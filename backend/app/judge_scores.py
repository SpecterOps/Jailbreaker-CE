"""Score contract for newly evaluated platform judges, independent of history."""

from __future__ import annotations


def validate_runtime_judge_score(value: object) -> int:
    """Require the prompt's integer 1–10 score without inventing a replacement."""
    if type(value) is not int or not 1 <= value <= 10:
        raise ValueError("judge score must be an integer between 1 and 10")
    return value
