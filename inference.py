"""
inference.py — Baseline inference script for Text-to-SQL Analyst environment.

Uses a local HuggingFace transformer model (via PyTorch) to generate SQL
from natural language questions. Falls back to OpenAI-compatible API if
MODEL_NAME points to a hosted endpoint.

Required env vars:
    API_BASE_URL  — Base URL of the OpenEnv environment (e.g. http://localhost:8000)
    MODEL_NAME    — HuggingFace model ID (e.g. tiiuae/falcon-rw-1b) or hosted model name
    HF_TOKEN      — HuggingFace token for gated models / API auth
"""

import json
import os
import sys
import re

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "tiiuae/falcon-rw-1b")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

TASKS = ["single_table_aggregation", "multi_table_join", "complex_analytics"]

SYSTEM_PROMPT = """You are an expert SQLite query writer.
Given a database schema and a natural language question, write ONE valid SQLite SQL query.
Rules:
- Output ONLY the raw SQL query
- No markdown, no backticks, no explanation
- Use proper SQLite syntax (DATE(), BETWEEN, CTEs with WITH)
- Always use table aliases for multi-table queries
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def schema_to_text(schema: dict) -> str:
    """Convert schema dict to CREATE TABLE statements."""
    lines = []
    for table, info in schema.items():
        col_defs = []
        for col, meta in info["columns"].items():
            defn = f"  {col} {meta['type']}"
            if meta.get("pk"):
                defn += " PRIMARY KEY"
            if meta.get("fk"):
                defn += f"  -- FK → {meta['fk']}"
            col_defs.append(defn)
        lines.append(f"CREATE TABLE {table} (\n" + ",\n".join(col_defs) + "\n);")
    return "\n\n".join(lines)


def sample_rows_to_text(sample_rows: dict) -> str:
    """Format sample rows as readable text."""
    lines = []
    for table, rows in sample_rows.items():
        lines.append(f"-- {table} sample rows:")
        for row in rows[:2]:
            lines.append(f"   {json.dumps(row)}")
    return "\n".join(lines)


def build_prompt(schema: dict, sample_rows: dict, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

Database Schema:
{schema_to_text(schema)}

Sample Data:
{sample_rows_to_text(sample_rows)}

Question: {question}

SQL Query:"""


def extract_sql(raw_output: str) -> str:
    """
    Extract clean SQL from model output.
    Handles markdown fences, preamble text, etc.
    """
    # Strip markdown fences
    raw_output = re.sub(r"```sql|```", "", raw_output, flags=re.IGNORECASE).strip()

    # If model echoed the prompt, take only the part after "SQL Query:"
    if "SQL Query:" in raw_output:
        raw_output = raw_output.split("SQL Query:")[-1].strip()

    # Take content up to the first blank line after the first SELECT/WITH
    lines = raw_output.splitlines()
    sql_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started and re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)", stripped, re.IGNORECASE):
            started = True
        if started:
            if stripped == "" and sql_lines:
                break
            sql_lines.append(line)

    return "\n".join(sql_lines).strip() if sql_lines else raw_output.strip()


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    """Load HuggingFace causal LM via PyTorch."""
    print(f"[INFO] Loading model: {MODEL_NAME}", file=sys.stderr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN or None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=300,
        do_sample=False,          # greedy — deterministic for RL baseline
        temperature=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("[INFO] Model loaded.", file=sys.stderr)
    return pipe


def generate_sql(pipe, prompt: str) -> str:
    """Run inference and return extracted SQL."""
    output = pipe(prompt)[0]["generated_text"]
    # Strip the prompt from the output (some models echo it)
    if output.startswith(prompt):
        output = output[len(prompt):]
    return extract_sql(output)


# ── Main RL loop ──────────────────────────────────────────────────────────────

def run():
    pipe = load_model()

    for task_id in TASKS:
        # Reset episode
        resp = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        env_state = resp.json()

        print(json.dumps({
            "type": "[START]",
            "task_id": task_id,
            "question": env_state["question"],
        }))

        prompt = build_prompt(
            schema=env_state["schema"],
            sample_rows=env_state["sample_rows"],
            question=env_state["question"],
        )

        done = False
        final_reward = 0.0
        final_feedback = ""

        while not done:
            sql = generate_sql(pipe, prompt)

            print(json.dumps({
                "type": "[STEP]",
                "task_id": task_id,
                "sql": sql,
            }))

            step_resp = requests.post(f"{API_BASE_URL}/step", json={"sql": sql})
            step_resp.raise_for_status()
            result = step_resp.json()

            final_reward = result["reward"]
            final_feedback = result["feedback"]
            done = result["done"]

            # If not done and got partial credit, refine the prompt with feedback
            if not done and result["reward"] < 1.0:
                state_resp = requests.get(f"{API_BASE_URL}/state")
                s = state_resp.json()
                prompt = build_prompt(
                    schema=s["schema"],
                    sample_rows=s["sample_rows"],
                    question=s["question"],
                ) + f"\n\n-- Previous attempt returned: {json.dumps(result['result'][:3])}\n-- Feedback: {result['feedback']}\n-- Refine the query:\nSQL Query:"

        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "reward": final_reward,
            "feedback": final_feedback,
        }))


if __name__ == "__main__":
    run()
