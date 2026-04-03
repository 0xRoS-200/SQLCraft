import re


def extract_tables(sql: str) -> set:
    """Extract table names referenced in FROM and JOIN clauses."""
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    return set(re.findall(pattern, sql, re.IGNORECASE))


def tables_match(agent_sql: str, ground_truth_sql: str) -> bool:
    """Check if agent used the same set of tables as ground truth."""
    return extract_tables(agent_sql) == extract_tables(ground_truth_sql)
