"""
Unit tests for all grader functions.
Run: python -m pytest tests/test_graders.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import tempfile
import pytest

from graders.sql_executor import execute_sql, is_valid_sql, exact_match, row_set_overlap, normalize_row
from graders.table_checker import extract_tables, tables_match
from graders.reward_computer import compute_reward

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northwind.db")


# ── sql_executor tests ─────────────────────────────────────────────────────────

class TestExecuteSQL:
    def test_valid_query(self):
        result = execute_sql("SELECT 1 AS n", DB_PATH)
        assert result == [(1,)]

    def test_syntax_error_returns_none(self):
        result = execute_sql("SELEKT * FORM orders", DB_PATH)
        assert result is None

    def test_valid_table_query(self):
        result = execute_sql("SELECT COUNT(*) FROM orders", DB_PATH)
        assert result is not None
        assert result[0][0] > 0

    def test_empty_result_is_not_none(self):
        result = execute_sql("SELECT * FROM orders WHERE 1=0", DB_PATH)
        assert result == []


class TestIsValidSQL:
    def test_valid(self):
        assert is_valid_sql("SELECT * FROM orders LIMIT 1", DB_PATH) is True

    def test_invalid(self):
        assert is_valid_sql("NOT VALID SQL AT ALL !!!", DB_PATH) is False


class TestNormalizeRow:
    def test_float_int_equal(self):
        assert normalize_row((100,)) == normalize_row((100.0,))

    def test_rounding(self):
        assert normalize_row((3.14159,)) == (3.14,)

    def test_string_stripped_lowered(self):
        assert normalize_row(("  HELLO  ",)) == ("hello",)

    def test_none_preserved(self):
        assert normalize_row((None,)) == (None,)


class TestExactMatch:
    def test_exact(self):
        assert exact_match([(1, "a")], [(1, "a")]) is True

    def test_order_insensitive(self):
        assert exact_match([(2, "b"), (1, "a")], [(1, "a"), (2, "b")]) is True

    def test_type_flexible(self):
        assert exact_match([(100,)], [(100.0,)]) is True

    def test_mismatch(self):
        assert exact_match([(1,)], [(2,)]) is False

    def test_none_agent(self):
        assert exact_match(None, [(1,)]) is False

    def test_both_empty(self):
        assert exact_match([], []) is True


class TestRowSetOverlap:
    def test_full_overlap(self):
        assert row_set_overlap([(1,), (2,)], [(1,), (2,)]) == 1.0

    def test_partial_overlap(self):
        score = row_set_overlap([(1,), (2,)], [(1,), (2,), (3,)])
        assert abs(score - 2/3) < 0.01

    def test_no_overlap(self):
        assert row_set_overlap([(9,)], [(1,), (2,)]) == 0.0

    def test_empty_ground_truth(self):
        assert row_set_overlap([], []) == 1.0

    def test_empty_agent(self):
        assert row_set_overlap([], [(1,)]) == 0.0


# ── table_checker tests ────────────────────────────────────────────────────────

class TestExtractTables:
    def test_simple_from(self):
        sql = "SELECT * FROM orders"
        assert extract_tables(sql) == {"orders"}

    def test_join(self):
        sql = "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id"
        assert extract_tables(sql) == {"orders", "customers"}

    def test_multiple_joins(self):
        sql = "SELECT * FROM a JOIN b ON a.id=b.id LEFT JOIN c ON b.id=c.id"
        assert extract_tables(sql) == {"a", "b", "c"}

    def test_case_insensitive(self):
        sql = "select * from Orders join Customers on 1=1"
        assert extract_tables(sql) == {"Orders", "Customers"}

    def test_cte(self):
        sql = "WITH cte AS (SELECT * FROM products) SELECT * FROM cte JOIN categories ON 1=1"
        tables = extract_tables(sql)
        assert "products" in tables
        assert "categories" in tables


class TestTablesMatch:
    def test_match(self):
        assert tables_match(
            "SELECT * FROM orders JOIN customers ON 1=1",
            "SELECT * FROM customers JOIN orders ON 1=1"
        ) is True

    def test_mismatch(self):
        assert tables_match(
            "SELECT * FROM orders",
            "SELECT * FROM orders JOIN customers ON 1=1"
        ) is False


# ── reward_computer tests ──────────────────────────────────────────────────────

GROUND_TRUTH_SQL = "SELECT ROUND(SUM(total_amount), 2) FROM orders WHERE status='completed' AND order_date >= '2024-01-01' AND order_date <= '2024-01-31'"


class TestComputeReward:
    def test_syntax_error_reward_0(self):
        reward, feedback = compute_reward("SELEKT BROKEN", [], GROUND_TRUTH_SQL, DB_PATH)
        assert reward == 0.0
        assert feedback == "syntax_error"

    def test_correct_gives_1(self):
        # Use ground truth SQL itself — should be exact match
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(GROUND_TRUTH_SQL)
        gt_result = c.fetchall()
        conn.close()

        reward, feedback = compute_reward(GROUND_TRUTH_SQL, gt_result, GROUND_TRUTH_SQL, DB_PATH)
        assert reward == 1.0
        assert feedback == "correct"

    def test_valid_wrong_table_gives_0_1(self):
        reward, feedback = compute_reward(
            "SELECT COUNT(*) FROM customers",
            [(999999.99,)],
            GROUND_TRUTH_SQL,
            DB_PATH
        )
        assert reward == 0.1

    def test_correct_tables_wrong_agg_gives_0_3(self):
        # Queries orders table (correct) but wrong aggregation
        reward, feedback = compute_reward(
            "SELECT COUNT(*) FROM orders WHERE status='completed'",
            [(999999.99,)],
            GROUND_TRUTH_SQL,
            DB_PATH
        )
        assert reward == 0.3

    def test_empty_result_when_answer_exists(self):
        reward, feedback = compute_reward(
            "SELECT * FROM orders WHERE 1=0",
            [(419197.17,)],
            GROUND_TRUTH_SQL,
            DB_PATH
        )
        assert reward == 0.1
        assert feedback == "no_results"

    def test_reward_always_float(self):
        reward, _ = compute_reward("SELECT 1", [], GROUND_TRUTH_SQL, DB_PATH)
        assert isinstance(reward, float)

    def test_reward_in_valid_range(self):
        for sql in [
            "BROKEN",
            "SELECT * FROM orders WHERE 1=0",
            "SELECT COUNT(*) FROM customers",
            "SELECT * FROM orders LIMIT 1",
        ]:
            reward, _ = compute_reward(sql, [(1,)], GROUND_TRUTH_SQL, DB_PATH)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for SQL: {sql}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
