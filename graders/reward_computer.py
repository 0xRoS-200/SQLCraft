from graders.sql_executor import execute_sql, is_valid_sql, exact_match, row_set_overlap
from graders.table_checker import extract_tables


def compute_reward(
    agent_sql: str,
    ground_truth_result: list,
    ground_truth_sql: str,
    db_path: str
) -> tuple[float, str]:
    """
    Compute reward using 4-tier partial credit system.

    Returns:
        (reward: float, feedback: str)

    Tiers:
        0.0  — Invalid SQL (syntax / runtime error)
        0.1  — Valid SQL but wrong tables or empty result
        0.3  — Correct tables but wrong grouping/filter/aggregation
        0.6  — ≥70% row overlap with ground truth
        1.0  — Exact match (order-insensitive, type-flexible)
    """
    # Tier 0: Invalid SQL
    agent_result = execute_sql(agent_sql, db_path)
    if agent_result is None:
        return 0.0, "syntax_error"

    # Tier 1: Valid SQL, empty result when answer should exist
    if not agent_result and ground_truth_result:
        return 0.1, "no_results"

    # Tier 4: Exact match (check early — avoid unnecessary partial scoring)
    if exact_match(agent_result, ground_truth_result):
        return 1.0, "correct"

    # Tier 3: Row overlap ≥ 70%
    overlap = row_set_overlap(agent_result, ground_truth_result)
    if overlap >= 0.7:
        return 0.6, "partial_match"

    # Tier 2: Correct tables used
    agent_tables = extract_tables(agent_sql)
    truth_tables = extract_tables(ground_truth_sql)
    if agent_tables == truth_tables:
        return 0.3, "wrong_aggregation"

    # Default: valid SQL but wrong approach
    return 0.1, "valid_sql"
