import sqlite3


def execute_sql(sql: str, db_path: str):
    """Execute SQL against SQLite DB. Returns list of tuples or None on error."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error:
        return None


def is_valid_sql(sql: str, db_path: str) -> bool:
    """Check if SQL is syntactically and semantically valid."""
    result = execute_sql(sql, db_path)
    return result is not None


def normalize_row(row: tuple) -> tuple:
    """Normalize a result row for type-flexible, case-insensitive comparison."""
    normalized = []
    for v in row:
        if v is None:
            normalized.append(None)
        elif isinstance(v, (int, float)):
            normalized.append(round(float(v), 2))
        else:
            normalized.append(str(v).strip().lower())
    return tuple(normalized)


def exact_match(agent_result: list, ground_truth_result: list) -> bool:
    """Order-insensitive, type-flexible exact match comparison."""
    if agent_result is None:
        return False
    agent_set = set(normalize_row(r) for r in agent_result)
    truth_set = set(normalize_row(r) for r in ground_truth_result)
    return agent_set == truth_set


def row_set_overlap(agent_result: list, ground_truth_result: list) -> float:
    """Fraction of ground-truth rows that appear in agent result (order-insensitive)."""
    if not ground_truth_result:
        return 1.0
    if not agent_result:
        return 0.0
    agent_set = set(normalize_row(r) for r in agent_result)
    truth_set = set(normalize_row(r) for r in ground_truth_result)
    return len(agent_set & truth_set) / len(truth_set)
