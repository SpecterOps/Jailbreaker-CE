from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from app.api.analysis_sql import classify_analysis_sql, normalize_analysis_sql

ConnectionFactory = Callable[[], Any]
DiscoverViews = Callable[[], list[str]]

_ANALYSIS_VIEWS: dict[str, str] = {
    "run_overview": "analysis_run_overview",
    "model_summary": "analysis_model_summary",
    "technique_summary": "analysis_technique_summary",
    "detection_summary": "analysis_detection_summary",
    "round_details": "analysis_round_details",
}

_ANALYSIS_QUERY_RELATIONS = frozenset(
    {
        *_ANALYSIS_VIEWS.values(),
        "evaluation_runs",
        "evaluation_run_detections",
        "evaluation_run_conversations",
        "evaluation_run_traces",
        "experiments",
        "experiment_datasets",
        "experiment_dataset_prompts",
        "experiment_cells",
    }
)
_ANALYSIS_STATEMENT_TIMEOUT_MS = 5000


class AnalysisQueryService:
    def connection(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Analysis requires psycopg and a reachable PostgreSQL database.") from exc

        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )
        return psycopg.connect(dsn, autocommit=True)

    def list_views(self, *, discover_views: DiscoverViews | None = None) -> list[dict[str, str]]:
        items = [{"key": key, "table": table} for key, table in _ANALYSIS_VIEWS.items()]
        known_tables = {item["table"] for item in items}
        items.extend(
            {"key": table, "table": table}
            for table in self._discover(discover_views)
            if table not in known_tables
        )
        return items

    def get_view(
        self,
        view_key: str,
        *,
        limit: int = 100,
        connection_factory: ConnectionFactory | None = None,
        discover_views: DiscoverViews | None = None,
    ) -> dict[str, object]:
        table = self.resolve_table(view_key, discover_views=discover_views)
        query = f"SELECT * FROM {table} LIMIT %s"

        with self._connect(connection_factory) as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY")
                try:
                    cur.execute(f"SET LOCAL statement_timeout = {_ANALYSIS_STATEMENT_TIMEOUT_MS}")
                    cur.execute(query, (max(1, min(limit, 500)),))
                    rows = cur.fetchall()
                    columns = [desc.name for desc in cur.description]
                finally:
                    cur.execute("ROLLBACK")

        items = [dict(zip(columns, row, strict=False)) for row in rows]
        return {"view": view_key, "table": table, "columns": columns, "items": items}

    def run_query(
        self,
        sql: str,
        *,
        limit: int = 200,
        connection_factory: ConnectionFactory | None = None,
        discover_views: DiscoverViews | None = None,
    ) -> dict[str, object]:
        normalized = self.normalize_sql(sql)
        mode = self.classify_sql(normalized, discover_views=discover_views)

        with self._connect(connection_factory) as conn:
            with conn.cursor() as cur:
                cur.execute("BEGIN READ ONLY")
                try:
                    cur.execute(f"SET LOCAL statement_timeout = {_ANALYSIS_STATEMENT_TIMEOUT_MS}")
                    cur.execute(normalized)
                    rows = cur.fetchmany(max(1, min(limit, 500)))
                    columns = [desc.name for desc in cur.description] if cur.description else []
                finally:
                    cur.execute("ROLLBACK")

        items = [dict(zip(columns, row, strict=False)) for row in rows]
        return {
            "mode": mode,
            "columns": columns,
            "items": items,
            "row_count": len(items),
        }

    def discover_views(self, *, connection_factory: ConnectionFactory | None = None) -> list[str]:
        query = """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
              AND table_name LIKE 'analysis!_%' ESCAPE '!'
            ORDER BY table_name
        """
        with self._connect(connection_factory) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return [row[0] for row in cur.fetchall()]

    def resolve_table(self, view_key: str, *, discover_views: DiscoverViews | None = None) -> str:
        table = _ANALYSIS_VIEWS.get(view_key, view_key)
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table):
            raise KeyError(f"Unknown analysis view: {view_key}")
        if table not in self._discover(discover_views) and table not in _ANALYSIS_VIEWS.values():
            raise KeyError(f"Unknown analysis view: {view_key}")
        return table

    def normalize_sql(self, sql: str) -> str:
        return normalize_analysis_sql(sql)

    def classify_sql(self, sql: str, *, discover_views: DiscoverViews | None = None) -> str:
        return classify_analysis_sql(sql, self.allowed_relations(discover_views=discover_views))

    def allowed_relations(self, *, discover_views: DiscoverViews | None = None) -> set[str]:
        discovered_views = {
            view
            for view in self._discover(discover_views)
            if re.fullmatch(r"analysis_[a-zA-Z0-9_]+", view)
        }
        return set(_ANALYSIS_QUERY_RELATIONS) | discovered_views

    def _connect(self, connection_factory: ConnectionFactory | None):
        if connection_factory is None:
            return self.connection()
        return connection_factory()

    def _discover(self, discover_views: DiscoverViews | None) -> list[str]:
        if discover_views is None:
            return self.discover_views()
        return discover_views()
