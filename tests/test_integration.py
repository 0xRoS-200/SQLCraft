"""
Integration tests for the full Text-to-SQL OpenEnv environment.

Tests the live FastAPI server end-to-end — requires the server to be running.

Usage:
    # Start server first:
    uvicorn main:app --port 8000

    # Then run:
    python tests/test_integration.py [--base-url http://localhost:8000]

    # Or against deployed HF Space:
    python tests/test_integration.py --base-url https://your-space.hf.space
"""

import argparse
import json
import sys
import requests

PASS = "\033[92m✅\033[0m"
FAIL = "\033[91m❌\033[0m"


def check(label: str, condition: bool, detail: str = ""):
    icon = PASS if condition else FAIL
    msg = f"  {icon}  {label}"
    if detail and not condition:
        msg += f"\n       → {detail}"
    print(msg)
    return condition


def run_integration_tests(base: str) -> bool:
    results = []

    # ── 1. Health check ────────────────────────────────────────────────────────
    print("\n[1] Health Check")
    r = requests.get(f"{base}/health", timeout=10)
    results.append(check("GET /health returns 200", r.status_code == 200))
    results.append(check("health body has status=ok", r.json().get("status") == "ok"))

    # ── 2. Reset contract ──────────────────────────────────────────────────────
    print("\n[2] Reset API Contract")
    for task_id in ["single_table_aggregation", "multi_table_join", "complex_analytics"]:
        r = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=10)
        results.append(check(f"POST /reset 200 for {task_id}", r.status_code == 200))
        d = r.json()
        for key in ["schema", "question", "sample_rows", "task_id"]:
            results.append(check(f"  reset() has key '{key}' for {task_id}", key in d,
                                 f"got keys: {list(d.keys())}"))
        results.append(check(f"  schema is non-empty for {task_id}", bool(d.get("schema"))))
        results.append(check(f"  question is a string for {task_id}",
                             isinstance(d.get("question"), str)))

    # ── 3. Invalid task_id ─────────────────────────────────────────────────────
    print("\n[3] Error Handling")
    r = requests.post(f"{base}/reset", json={"task_id": "nonexistent_task"}, timeout=10)
    results.append(check("reset() with bad task_id returns 400", r.status_code == 400))

    r = requests.get(f"{base}/state", timeout=10)
    # After a valid reset above, state should work
    requests.post(f"{base}/reset", json={"task_id": "single_table_aggregation"})
    r = requests.get(f"{base}/state", timeout=10)
    results.append(check("GET /state returns 200 after reset", r.status_code == 200))
    s = r.json()
    for key in ["schema", "question", "sample_rows", "last_sql", "last_result",
                "attempt_number", "max_attempts"]:
        results.append(check(f"  state() has key '{key}'", key in s))

    # ── 4. Step contract ───────────────────────────────────────────────────────
    print("\n[4] Step API Contract")
    requests.post(f"{base}/reset", json={"task_id": "single_table_aggregation"})
    r = requests.post(f"{base}/step", json={"sql": "SELECT 1"}, timeout=10)
    results.append(check("POST /step returns 200", r.status_code == 200))
    s = r.json()
    for key in ["result", "reward", "done", "feedback"]:
        results.append(check(f"  step() has key '{key}'", key in s))
    results.append(check("  reward is float", isinstance(s.get("reward"), float),
                         f"got type: {type(s.get('reward')).__name__}"))
    results.append(check("  reward in [0.0, 1.0]",
                         0.0 <= s.get("reward", -1) <= 1.0,
                         f"got: {s.get('reward')}"))
    results.append(check("  done is bool", isinstance(s.get("done"), bool)))
    results.append(check("  feedback is string", isinstance(s.get("feedback"), str)))

    # ── 5. Reward tiers — Task 1 ───────────────────────────────────────────────
    print("\n[5] Reward Tiers — Task 1 (single_table_aggregation)")

    def reset_and_step(task_id, sql):
        requests.post(f"{base}/reset", json={"task_id": task_id})
        return requests.post(f"{base}/step", json={"sql": sql}, timeout=10).json()

    # Tier 0: syntax error
    s = reset_and_step("single_table_aggregation", "SELEKT BROKEN SQL !!!")
    results.append(check("Tier 0: syntax error → reward=0.0",
                         s["reward"] == 0.0, f"got {s['reward']}"))
    results.append(check("Tier 0: feedback='syntax_error'",
                         s["feedback"] == "syntax_error", f"got '{s['feedback']}'"))

    # Tier 1: valid SQL, empty result
    s = reset_and_step("single_table_aggregation", "SELECT * FROM orders WHERE 1=0")
    results.append(check("Tier 1: empty result → reward=0.1",
                         s["reward"] == 0.1, f"got {s['reward']}"))

    # Tier 2: valid SQL, wrong table
    s = reset_and_step("single_table_aggregation", "SELECT COUNT(*) FROM customers")
    results.append(check("Tier 1/2: wrong table → reward≤0.1",
                         s["reward"] <= 0.1, f"got {s['reward']}"))

    # Tier 2: correct table, wrong aggregation
    s = reset_and_step("single_table_aggregation",
                       "SELECT COUNT(*) FROM orders WHERE status='completed'")
    results.append(check("Tier 2: correct table, wrong agg → reward=0.3",
                         s["reward"] == 0.3, f"got {s['reward']}"))

    # Tier 4: exact match
    correct_sql = ("SELECT ROUND(SUM(total_amount), 2) AS total_revenue FROM orders "
                   "WHERE status='completed' AND order_date >= '2024-01-01' "
                   "AND order_date <= '2024-01-31'")
    s = reset_and_step("single_table_aggregation", correct_sql)
    results.append(check("Tier 4: correct SQL → reward=1.0",
                         s["reward"] == 1.0, f"got {s['reward']}"))
    results.append(check("Tier 4: feedback='correct'",
                         s["feedback"] == "correct", f"got '{s['feedback']}'"))
    results.append(check("Tier 4: done=True on correct answer",
                         s["done"] is True))

    # ── 6. Multi-attempt episode ───────────────────────────────────────────────
    print("\n[6] Multi-Attempt Episode Flow")
    requests.post(f"{base}/reset", json={"task_id": "single_table_aggregation"})

    # Attempt 1: wrong
    s1 = requests.post(f"{base}/step", json={"sql": "SELECT 1"}, timeout=10).json()
    results.append(check("Attempt 1: done=False on wrong answer",
                         s1["done"] is False, f"done={s1['done']}"))

    # Attempt 2: wrong
    s2 = requests.post(f"{base}/step", json={"sql": "SELECT 2"}, timeout=10).json()
    results.append(check("Attempt 2: done=False", s2["done"] is False))

    # Attempt 3: wrong — should be done=True because max_attempts=3
    s3 = requests.post(f"{base}/step", json={"sql": "SELECT 3"}, timeout=10).json()
    results.append(check("Attempt 3: done=True after max_attempts",
                         s3["done"] is True, f"done={s3['done']}"))

    # Calling step again after done should error
    r = requests.post(f"{base}/step", json={"sql": "SELECT 4"}, timeout=10)
    results.append(check("Step after done returns 400", r.status_code == 400))

    # ── 7. Reward determinism ──────────────────────────────────────────────────
    print("\n[7] Determinism — Same SQL Must Give Same Reward")
    sql = "SELECT ROUND(SUM(total_amount), 2) FROM orders WHERE status='completed'"
    rewards = []
    for _ in range(3):
        requests.post(f"{base}/reset", json={"task_id": "single_table_aggregation"})
        s = requests.post(f"{base}/step", json={"sql": sql}, timeout=10).json()
        rewards.append(s["reward"])
    results.append(check("Same SQL always returns same reward",
                         len(set(rewards)) == 1, f"got: {rewards}"))

    # ── 8. Task 3 (hard) end-to-end ───────────────────────────────────────────
    print("\n[8] Task 3 (Hard) — Complex Analytics End-to-End")
    hard_sql = """
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
""".strip()

    s = reset_and_step("complex_analytics", hard_sql)
    results.append(check("Task 3 ground truth SQL → reward=1.0",
                         s["reward"] == 1.0, f"got {s['reward']}"))
    results.append(check("Task 3 result is non-empty list",
                         isinstance(s["result"], list) and len(s["result"]) > 0,
                         f"got: {s['result']}"))

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'─'*50}")
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print(f"  {PASS} ALL INTEGRATION TESTS PASSED — ready to submit!")
    else:
        print(f"  {FAIL} {total - passed} check(s) failed — fix before submitting.")
    print(f"{'─'*50}\n")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the running environment server")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    print(f"Running integration tests against: {base}")

    try:
        success = run_integration_tests(base)
        sys.exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print(f"\n{FAIL} Could not connect to {base}")
        print("  Make sure the server is running: uvicorn main:app --port 8000")
        sys.exit(1)
