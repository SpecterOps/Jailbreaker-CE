from __future__ import annotations

import re

__all__ = ["classify_analysis_sql", "normalize_analysis_sql"]


_ANALYSIS_BLOCKED_KEYWORDS = frozenset(
    {
        "alter",
        "analyze",
        "call",
        "copy",
        "create",
        "delete",
        "do",
        "drop",
        "execute",
        "grant",
        "insert",
        "into",
        "listen",
        "lock",
        "merge",
        "notify",
        "refresh",
        "reindex",
        "reset",
        "revoke",
        "set",
        "truncate",
        "update",
        "vacuum",
        "with",
    }
)
_ANALYSIS_BLOCKED_FUNCTIONS = frozenset(
    {
        "current_setting",
        "dblink",
        "dblink_exec",
        "lo_export",
        "lo_import",
        "pg_ls_dir",
        "pg_read_binary_file",
        "pg_read_file",
        "pg_sleep",
        "pg_stat_file",
        "set_config",
    }
)
_ANALYSIS_ALLOWED_FUNCTIONS = frozenset(
    {
        "avg",
        "coalesce",
        "count",
        "date_trunc",
        "lower",
        "max",
        "min",
        "nullif",
        "round",
        "sum",
        "upper",
    }
)
_ANALYSIS_ALLOWED_CALL_KEYWORDS = frozenset({"in"})


def _analysis_sql_tokens(sql: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    length = len(sql)
    while i < length:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "'":
            i += 1
            while i < length:
                if sql[i] == "'" and i + 1 < length and sql[i + 1] == "'":
                    i += 2
                    continue
                if sql[i] == "'":
                    i += 1
                    break
                i += 1
            continue
        if ch == '"':
            i += 1
            quoted: list[str] = []
            while i < length:
                if sql[i] == '"' and i + 1 < length and sql[i + 1] == '"':
                    quoted.append('"')
                    i += 2
                    continue
                if sql[i] == '"':
                    i += 1
                    break
                quoted.append(sql[i])
                i += 1
            if quoted:
                tokens.append("".join(quoted).lower())
            continue
        if ch == "-" and i + 1 < length and sql[i + 1] == "-":
            i += 2
            while i < length and sql[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and i + 1 < length and sql[i + 1] == "*":
            i += 2
            while i + 1 < length and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i = min(length, i + 2)
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < length and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            tokens.append(sql[start:i].lower())
            continue
        if ch.isdigit():
            while i < length and (sql[i].isdigit() or sql[i] == "."):
                i += 1
            tokens.append("#")
            continue
        tokens.append(ch)
        i += 1
    return tokens


def _analysis_identifier(token: str) -> bool:
    return bool(re.fullmatch(r"[a-z_][a-z0-9_$]*", token))


def _analysis_relation_at(tokens: list[str], index: int) -> tuple[str, int]:
    if index >= len(tokens) or not _analysis_identifier(tokens[index]):
        raise ValueError("Analysis SQL must name an approved table or view after FROM/JOIN.")
    parts = [tokens[index]]
    index += 1
    while index + 1 < len(tokens) and tokens[index] == ".":
        if not _analysis_identifier(tokens[index + 1]):
            raise ValueError("Analysis SQL has an invalid table or view reference.")
        parts.append(tokens[index + 1])
        index += 2
    if len(parts) == 2:
        schema, relation = parts
        if schema != "public":
            raise ValueError("Analysis SQL can only read approved public reporting tables/views.")
        return relation, index
    if len(parts) > 2:
        raise ValueError("Analysis SQL can only read approved public reporting tables/views.")
    return parts[0], index


def _analysis_referenced_relations(tokens: list[str]) -> set[str]:
    relations: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in {"from", "join"}:
            index += 1
            continue
        index += 1
        while index < len(tokens) and tokens[index] in {"only"}:
            index += 1
        if index < len(tokens) and tokens[index] == "(":
            raise ValueError("Subqueries are not allowed in analysis SQL.")
        relation, index = _analysis_relation_at(tokens, index)
        relations.add(relation)
    return relations


def normalize_analysis_sql(sql: str) -> str:
    normalized = (sql or "").strip()
    if not normalized:
        raise ValueError("SQL is required.")
    tokens = _analysis_sql_tokens(normalized)
    if not tokens:
        raise ValueError("SQL is required.")
    if tokens.count(";") > 1 or (";" in tokens and tokens[-1] != ";"):
        raise ValueError("Only a single SQL statement is allowed.")
    return normalized.rstrip(";").strip()


def classify_analysis_sql(sql: str, allowed_relations: set[str]) -> str:
    tokens = _analysis_sql_tokens(sql)
    first = tokens[0] if tokens else ""
    if first == "create" and "view" in tokens:
        raise ValueError("Creating analysis views from the SQL endpoint is disabled.")
    if first == "with":
        raise ValueError("Only direct SELECT queries are supported; WITH/CTE is disabled.")
    if first != "select":
        raise ValueError("Only read-only SELECT queries are supported.")
    _validate_analysis_select_tokens(tokens, allowed_relations)
    return "query"


def _validate_analysis_select_tokens(tokens: list[str], allowed_relations: set[str]) -> None:
    if "select" in tokens[1:]:
        raise ValueError("Subqueries and compound SELECT statements are not allowed.")
    blocked = sorted(set(tokens) & _ANALYSIS_BLOCKED_KEYWORDS)
    if blocked:
        raise ValueError(f"Analysis SQL contains blocked keyword: {blocked[0]}.")
    for index, token in enumerate(tokens[:-1]):
        if not _analysis_identifier(token) or tokens[index + 1] != "(":
            continue
        if token in _ANALYSIS_ALLOWED_CALL_KEYWORDS:
            continue
        if token in _ANALYSIS_BLOCKED_FUNCTIONS or token not in _ANALYSIS_ALLOWED_FUNCTIONS:
            raise ValueError(f"Analysis SQL cannot call unapproved function: {token}.")
    relations = _analysis_referenced_relations(tokens)
    if not relations:
        raise ValueError("Analysis SQL must read from approved reporting tables/views.")
    unknown = sorted(relations - allowed_relations)
    if unknown:
        raise ValueError(
            "Analysis SQL can only read approved reporting tables/views: "
            + ", ".join(unknown)
        )
