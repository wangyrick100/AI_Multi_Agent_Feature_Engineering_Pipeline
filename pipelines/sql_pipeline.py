"""
SQL Pipeline
─────────────
Assembles BigQuery-compatible SQL (or generic ANSI SQL) for all selected
features into a single CREATE TABLE / SELECT statement.

Usage
-----
    from pipelines.sql_pipeline import SQLPipeline

    pipe = SQLPipeline(selected_definitions, source_table="raw_events")
    sql = pipe.generate()
    print(sql)
    pipe.save_sql("features.sql")
"""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# SQL dialect constants
BIGQUERY = "bigquery"
SNOWFLAKE = "snowflake"
POSTGRES = "postgres"
ANSI = "ansi"


class SQLPipeline:
    """
    Generates a monolithic SQL SELECT that computes all selected features.

    Parameters
    ----------
    selected_definitions  List of FeatureDefinition objects or dicts.
    source_table          Source table name (used in FROM clause).
    entity_col            Primary entity column (e.g. user_id).
    ts_col                Timestamp column name.
    dialect               SQL dialect: 'bigquery', 'snowflake', 'postgres', 'ansi'.
    """

    def __init__(
        self,
        selected_definitions: Optional[list] = None,
        source_table: str = "raw_events",
        entity_col: str = "user_id",
        ts_col: str = "event_timestamp",
        dialect: str = BIGQUERY,
    ) -> None:
        self.source_table = source_table
        self.entity_col = entity_col
        self.ts_col = ts_col
        self.dialect = dialect
        self._defs: List[Dict] = []
        if selected_definitions:
            self._load(selected_definitions)

    # ── Load definitions ─────────────────────────────────────────────────────

    def _load(self, defs: list) -> None:
        for d in defs:
            if hasattr(d, "model_dump"):
                self._defs.append(d.model_dump())
            elif isinstance(d, dict):
                self._defs.append(d)
            else:
                self._defs.append(vars(d))

    # ── SQL generation ───────────────────────────────────────────────────────

    def generate(self, create_table: Optional[str] = None) -> str:
        """
        Return the full SQL text.

        Parameters
        ----------
        create_table  If provided, wraps output in CREATE OR REPLACE TABLE ... AS (...)
        """
        columns: List[str] = ["*"]  # always include source columns

        for d in self._defs:
            sql_expr = d.get("sql_code", "")
            name = d.get("name", d.get("feature_id", "unknown"))
            columns.append(f"  {self._render_expression(sql_expr, name)}")

        col_block = ",\n".join(columns)
        select_sql = f"SELECT\n{col_block}\nFROM {self._quote_table(self.source_table)}"

        if create_table:
            prefix = (
                f"CREATE OR REPLACE TABLE {self._quote_table(create_table)} AS\n"
                if self.dialect in (BIGQUERY, SNOWFLAKE)
                else f"CREATE TABLE {self._quote_table(create_table)} AS\n"
            )
            return prefix + "(\n" + textwrap.indent(select_sql, "  ") + "\n)"

        return select_sql

    def generate_ctes(self) -> str:
        """
        Generate a modular SQL script with one CTE per feature category.
        Useful for readability and incremental materialisation.
        """
        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for d in self._defs:
            cat = str(d.get("category", "other"))
            by_category.setdefault(cat, []).append(d)

        cte_blocks: List[str] = []
        final_cols: List[str] = ["base.*"]

        for cat, defs in by_category.items():
            cte_name = f"cte_{cat.lower().replace(' ', '_')}"
            inner_cols = [self.entity_col]
            for d in defs:
                sql_expr = d.get("sql_code", "").strip().rstrip(",")
                name = d.get("name", "unknown")
                inner_cols.append(f"  {self._render_expression(sql_expr, name)}")
            cte_block = (
                f"{cte_name} AS (\n"
                f"  SELECT\n"
                + ",\n    ".join(inner_cols)
                + f"\n  FROM {self._quote_table(self.source_table)}\n)"
            )
            cte_blocks.append(cte_block)

            for d in defs:
                name = d.get("name", "unknown")
                final_cols.append(f"{cte_name}.{self._quote(name)}")

        joins = "\n".join(
            f"LEFT JOIN {cte_name} USING ({self.entity_col})"
            for cte_name in by_category
        )

        cte_sql = "WITH\n" + ",\n\n".join(cte_blocks)
        select_sql = (
            "\nSELECT\n  "
            + ",\n  ".join(final_cols)
            + f"\nFROM base\n{joins}"
        )
        return cte_sql + select_sql

    def generate_incremental(
        self,
        lookback_days: int = 90,
        incremental_ts_param: str = "@run_ts",
    ) -> str:
        """
        Generate an incremental SQL that only processes new rows since
        `incremental_ts_param`. Joins back to a features table for
        window aggregations with a configurable lookback.
        """
        ts_expr = f"TIMESTAMP_SUB({incremental_ts_param}, INTERVAL {lookback_days} DAY)"
        base_filter = (
            f"WHERE {self.ts_col} >= {ts_expr}\n  AND {self.ts_col} < {incremental_ts_param}"
        )
        return (
            f"-- Incremental (lookback={lookback_days}d, run_at={incremental_ts_param})\n"
            + self.generate()
            + "\n"
            + base_filter
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_sql(self, path: str, create_table: Optional[str] = None) -> None:
        sql = self.generate(create_table=create_table)
        with open(path, "w", encoding="utf-8") as f:
            f.write(sql)
        logger.info("SQL pipeline saved → %s  (%d features)", path, len(self._defs))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _quote(self, name: str) -> str:
        if self.dialect == BIGQUERY:
            return f"`{name}`"
        return f'"{name}"'

    def _render_expression(self, sql_expr: str, name: str) -> str:
        cleaned = (sql_expr or "").strip().rstrip(",")
        if not cleaned:
            return f"NULL AS {self._quote(name)}"

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        non_comment_lines = [line for line in lines if not line.startswith("--")]
        if not non_comment_lines:
            return f"NULL AS {self._quote(name)}"

        expr = "\n  ".join(non_comment_lines)
        if " AS " in expr.upper():
            return expr
        return f"{expr} AS {self._quote(name)}"

    def _quote_table(self, name: str) -> str:
        if "." in name:  # already qualified
            return name
        return self._quote(name)

    def __len__(self) -> int:
        return len(self._defs)

    def __repr__(self) -> str:
        return f"SQLPipeline(features={len(self._defs)}, table='{self.source_table}', dialect='{self.dialect}')"
