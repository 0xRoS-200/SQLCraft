---
title: Text-to-SQL Analyst
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Text-to-SQL Analyst — OpenEnv Environment

> **OpenEnv Hackathon Round 1** | Meta × Hugging Face | April 2026

An AI agent receives a database schema + natural language business question and must generate correct SQLite SQL to answer it. Graded deterministically by executing the SQL and comparing results to ground truth.

---

## Schema Diagram

```
┌─────────────────────────┐        ┌──────────────────────────┐
│        customers        │        │          orders           │
├─────────────────────────┤        ├──────────────────────────┤
│ customer_id  PK INTEGER │◄───────│ customer_id  FK INTEGER  │
│ customer_name    TEXT   │        │ order_id     PK INTEGER  │
│ city             TEXT   │        │ total_amount    REAL     │
│ signup_date      DATE   │        │ order_date      DATE     │
│ tier             TEXT   │        │ status          TEXT     │
└─────────────────────────┘        └──────────┬───────────────┘
                                              │
                    ┌─────────────────────────┴──────────────┐
                    │                                        │
          ┌─────────▼──────────┐               ┌────────────▼────────────┐
          │    order_items     │               │         returns          │
          ├────────────────────┤               ├──────────────────────────┤
          │ item_id   PK INT   │               │ return_id   PK INTEGER   │
          │ order_id  FK INT   │               │ order_id    FK INTEGER   │
          │ product_id FK INT  │               │ product_id  FK INTEGER   │
          │ quantity   INTEGER │               │ return_date DATE         │
          │ unit_price REAL    │               │ reason      TEXT         │
          └─────────┬──────────┘               └──────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │      products      │
          ├────────────────────┤
          │ product_id  PK INT │
          │ product_name TEXT  │
          │ category_id FK INT │◄────┐
          │ cost_price  REAL   │     │
          └────────────────────┘     │
                                     │
                          ┌──────────┴─────────┐
                          │      categories    │
                          ├────────────────────┤
                          │ category_id PK INT │
                          │ category_name TEXT │
                          │ department   TEXT  │
                          └────────────────────┘
```

---

## Tasks

### Task 1 — Easy: Single Table Aggregation
**Question:** *"What is the total revenue from completed orders placed in January 2024?"*

**Expected SQL:**
```sql
SELECT ROUND(SUM(total_amount), 2) AS total_revenue
FROM orders
WHERE status = 'completed'
  AND order_date >= '2024-01-01'
  AND order_date <= '2024-01-31'
```

**Baseline score:** `1.0` (exact match)

---

### Task 2 — Medium: Multi-Table JOIN with Grouping
**Question:** *"List the top 5 customers by total spend in the last 6 months, along with their city and number of orders."*

**Expected SQL:**
```sql
SELECT c.customer_name, c.city,
       COUNT(o.order_id) AS order_count,
       ROUND(SUM(o.total_amount), 2) AS total_spend
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2025-10-01'
  AND o.status = 'completed'
GROUP BY c.customer_id, c.customer_name, c.city
ORDER BY total_spend DESC
LIMIT 5
```

**Baseline score:** `0.6` (partial — model gets right customers, wrong order)

---

### Task 3 — Hard: 5-Table Analytics with CTEs
**Question:** *"Which product categories had a return rate above 20% in Q4 2023, and what was their average order value?"*

**Expected SQL:**
```sql
WITH category_stats AS (
    SELECT cat.category_name,
           COUNT(DISTINCT oi.item_id) AS total_items_sold,
           COUNT(DISTINCT r.return_id) AS total_returns,
           AVG(o.total_amount) AS avg_order_value
    FROM categories cat
    JOIN products p ON p.category_id = cat.category_id
    JOIN order_items oi ON oi.product_id = p.product_id
    JOIN orders o ON o.order_id = oi.order_id
    LEFT JOIN returns r ON r.order_id = o.order_id AND r.product_id = p.product_id
    WHERE o.order_date BETWEEN '2023-10-01' AND '2023-12-31'
      AND o.status = 'completed'
    GROUP BY cat.category_id, cat.category_name
)
SELECT category_name,
       ROUND(100.0 * total_returns / total_items_sold, 2) AS return_rate_pct,
       ROUND(avg_order_value, 2) AS avg_order_value
FROM category_stats
WHERE total_items_sold > 0
  AND CAST(total_returns AS FLOAT) / total_items_sold > 0.20
ORDER BY return_rate_pct DESC
```

**Baseline score:** `1.0` (exact match with CTE)

---

## Reward Function

| Score | Label | Description |
|-------|-------|-------------|
| `0.0` | `syntax_error` | SQL fails to execute |
| `0.1` | `valid_sql` / `no_results` | Valid SQL, wrong tables or empty result |
| `0.3` | `wrong_aggregation` | Correct tables, wrong grouping/filter |
| `0.6` | `partial_match` | ≥70% row overlap with ground truth |
| `1.0` | `correct` | Exact match (order-insensitive, type-flexible) |

---

## Action & Observation Spaces

**Action:** A single SQLite SQL string.

**Observation (from `/reset`):**
```json
{
  "schema": { "table_name": { "columns": { "col": { "type": "...", "pk": true } } } },
  "question": "Natural language question",
  "sample_rows": { "table_name": [{ "col": "value" }] },
  "task_id": "single_table_aggregation"
}
```

**Step response:**
```json
{
  "result": [[row1], [row2]],
  "reward": 1.0,
  "done": true,
  "feedback": "correct"
}
```

---

## Setup & Run

### Local

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Run inference baseline
export API_BASE_URL=http://localhost:8000
export MODEL_NAME=tiiuae/falcon-rw-1b
export HF_TOKEN=your_hf_token_here
python inference.py
```

### Docker

```bash
docker build -t text-to-sql-analyst .
docker run -p 8000:8000 \
  -e API_BASE_URL=http://localhost:8000 \
  -e MODEL_NAME=tiiuae/falcon-rw-1b \
  -e HF_TOKEN=your_token \
  text-to-sql-analyst
```

### Run Tests

```bash
python -m pytest tests/test_graders.py -v
# Expected: 35 passed
```

---

## Baseline Scores

| Task | Difficulty | Baseline Reward | Feedback |
|------|-----------|----------------|---------|
| single_table_aggregation | Easy | `1.0` | correct |
| multi_table_join | Medium | `0.6` | partial_match |
| complex_analytics | Hard | `1.0` | correct |

> Baseline uses `tiiuae/falcon-rw-1b` via PyTorch on CPU. Upgrade `MODEL_NAME` to a code-tuned model (e.g., `defog/sqlcoder-7b-2`) for higher scores on the medium task.

---

## Tech Stack

- **FastAPI** — OpenEnv REST API server
- **SQLite** — Zero-infrastructure database (built into Python)
- **PyTorch + HuggingFace Transformers** — Local LLM inference for SQL generation
- **OpenCV** — Included for visual schema rendering capability
- **Pydantic v2** — Typed request/response models

---

## File Structure

```
text-to-sql-analyst/
├── main.py                        # FastAPI app — reset/step/state endpoints
├── inference.py                   # Baseline PyTorch LLM inference script
├── openenv.yaml                   # OpenEnv spec
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   ├── northwind.db               # SQLite database
│   └── ground_truth/
│       ├── task1.json
│       ├── task2.json
│       └── task3.json
├── graders/
│   ├── sql_executor.py            # execute_sql, exact_match, row_set_overlap
│   ├── table_checker.py           # extract_tables, tables_match
│   └── reward_computer.py         # compute_reward() — 4-tier function
└── tests/
    └── test_graders.py            # 35 unit tests — all passing
```
