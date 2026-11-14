import csv
import json
from pathlib import Path

DATA_FILE = Path("exports/account_health.csv")
OUTPUT_FILE = Path("build/churn_risk_summary.json")


def load_accounts(path: Path = DATA_FILE) -> list[dict]:
    # TODO: parse rows with account_id, company, mrr, health_score, risk_score, open_tickets
    raise NotImplementedError


def rank_accounts(accounts: list[dict]) -> list[dict]:
    # TODO: prioritize accounts with high risk, low health, and high MRR
    raise NotImplementedError


def write_report(rows: list[dict], path: Path = OUTPUT_FILE) -> None:
    # TODO: create the output directory and write deterministic JSON
    raise NotImplementedError


def main() -> None:
    accounts = load_accounts()
    write_report(rank_accounts(accounts))


if __name__ == "__main__":
    main()
