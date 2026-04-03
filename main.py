import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graders.reward_computer import compute_reward
from graders.sql_executor import execute_sql, is_valid_sql

app = FastAPI(title="Text-to-SQL Analyst Environment", version="1.0.0")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DB_PATH = str(BASE_DIR / "data" / "northwind.db")
GT_DIR = BASE_DIR / "data" / "ground_truth"

# ── Task config ────────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "single_table_aggregation": {
        "tables": ["orders"],
        "difficulty": "easy",
        "max_attempts": 3,
    },
    "multi_table_join": {
        "tables": ["customers", "orders"],
        "difficulty": "medium",
        "max_attempts": 3,
    },
    "complex_analytics": {
        "tables": ["orders", "order_items", "returns", "products", "categories"],
        "difficulty": "hard",
        "max_attempts": 3,
    },
}

# ── In-memory episode state ────────────────────────────────────────────────────
_state: dict[str, Any] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_ground_truth(task_id: str) -> dict:
    path = GT_DIR / f"{_task_file(task_id)}.json"
    with open(path) as f:
        return json.load(f)


def _task_file(task_id: str) -> str:
    mapping = {
        "single_table_aggregation": "task1",
        "multi_table_join": "task2",
        "complex_analytics": "task3",
    }
    return mapping[task_id]


def _get_schema(tables: list[str]) -> dict:
    """Build schema dict from SQLite introspection."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    schema = {}

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols_info = cursor.fetchall()
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fk_info = {row[3]: f"{row[2]}.{row[4]}" for row in cursor.fetchall()}

        columns = {}
        for col in cols_info:
            _, name, col_type, _, _, pk = col
            meta: dict[str, Any] = {"type": col_type or "TEXT"}
            if pk:
                meta["pk"] = True
            if name in fk_info:
                meta["fk"] = fk_info[name]
            columns[name] = meta
        schema[table] = {"columns": columns}

    conn.close()
    return schema


def _get_sample_rows(tables: list[str], n: int = 3) -> dict:
    """Fetch n sample rows per table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    samples = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table} LIMIT {n}")
        rows = cursor.fetchall()
        cursor.execute(f"PRAGMA table_info({table})")
        col_names = [r[1] for r in cursor.fetchall()]
        samples[table] = [dict(zip(col_names, row)) for row in rows]
    conn.close()
    return samples


# ── Request / Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str


class StepRequest(BaseModel):
    sql: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    task_id = req.task_id
    if task_id not in TASK_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}. "
                            f"Valid: {list(TASK_CONFIG.keys())}")

    cfg = TASK_CONFIG[task_id]
    gt = _load_ground_truth(task_id)

    _state.clear()
    _state.update({
        "task_id": task_id,
        "question": gt["question"],
        "ground_truth_sql": gt["sql"],
        "ground_truth_result": [tuple(r) for r in gt["expected_result"]],
        "schema": _get_schema(cfg["tables"]),
        "sample_rows": _get_sample_rows(cfg["tables"]),
        "last_sql": None,
        "last_result": None,
        "attempt_number": 0,
        "max_attempts": cfg["max_attempts"],
        "done": False,
    })

    return {
        "schema": _state["schema"],
        "question": _state["question"],
        "sample_rows": _state["sample_rows"],
        "task_id": task_id,
    }


@app.post("/step")
def step(req: StepRequest):
    if not _state:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if _state["done"]:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")

    sql = req.sql.strip()
    _state["attempt_number"] += 1
    _state["last_sql"] = sql

    reward, feedback = compute_reward(
        agent_sql=sql,
        ground_truth_result=_state["ground_truth_result"],
        ground_truth_sql=_state["ground_truth_sql"],
        db_path=DB_PATH,
    )

    result = execute_sql(sql, DB_PATH) or []
    _state["last_result"] = result

    done = (reward == 1.0) or (_state["attempt_number"] >= _state["max_attempts"])
    _state["done"] = done

    return {
        "result": result,
        "reward": reward,
        "done": done,
        "feedback": feedback,
    }


@app.get("/state")
def state():
    if not _state:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return {
        "schema": _state["schema"],
        "question": _state["question"],
        "sample_rows": _state["sample_rows"],
        "last_sql": _state["last_sql"],
        "last_result": _state["last_result"],
        "attempt_number": _state["attempt_number"],
        "max_attempts": _state["max_attempts"],
    }
