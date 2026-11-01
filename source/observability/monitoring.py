"""Local, privacy-preserving storage and aggregation of SensAI metrics."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import FeatureConfig, MonitoringConfig


@dataclass(frozen=True)
class MetricsFilter:
    """Optional filters supported by the local metrics dashboard."""

    start_date: str | None = None
    end_date: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    feature: str | None = None
    tool_name: str | None = None
    error_category: str | None = None


@dataclass(frozen=True)
class DashboardMetrics:
    """Aggregated, non-sensitive metrics displayed by the CLI dashboard."""

    model_calls: int
    average_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    rag_calls: int
    input_tokens: int
    output_tokens: int
    tool_calls: int
    tool_successes: int
    tool_failures: int
    daily_calls: list[tuple[str, int]]


class MonitoringStore:
    """Persist interaction metadata locally without storing request or response content."""

    def __init__(self, database_path: str | Path, retention_days: int) -> None:
        self.database_path = Path(database_path)
        self.retention_days = retention_days
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def enabled(self) -> bool:
        return True

    def record_model_interaction(
        self,
        *,
        model: str,
        features: FeatureConfig,
        prompt_version: str,
        status: str,
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
        rag_chunk_count: int,
        error_category: str | None = None,
    ) -> int:
        """Store safe metadata for a single request; no prompt or response is retained."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_interactions (
                    timestamp, component, model, prompt_version, status, latency_ms,
                    input_tokens, output_tokens, rag_chunk_count,
                    rag_enabled, tools_enabled, memory_enabled, error_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _timestamp(),
                    "model",
                    model,
                    prompt_version,
                    status,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    rag_chunk_count,
                    int(features.rag_enabled),
                    int(features.tools_enabled),
                    int(features.memory_enabled),
                    error_category,
                ),
            )
            return int(cursor.lastrowid)

    def record_tool_interaction(
        self,
        *,
        model_interaction_id: int,
        tool_name: str,
        status: str,
        latency_ms: float,
    ) -> None:
        """Store a tool outcome without its arguments, result, or other content."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_interactions (
                    timestamp, component, model_interaction_id, tool_name, status, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_timestamp(), "tool", model_interaction_id, tool_name, status, latency_ms),
            )

    def dashboard(self, filters: MetricsFilter = MetricsFilter()) -> DashboardMetrics:
        """Aggregate persisted metrics for a compact local dashboard."""
        where, parameters = self._interaction_filter(filters)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, timestamp, status, latency_ms, input_tokens, output_tokens, rag_chunk_count "
                f"FROM model_interactions {where}",
                parameters,
            ).fetchall()
            interaction_ids = [int(row[0]) for row in rows]
            tool_rows = self._tool_rows(connection, interaction_ids, filters.tool_name)

        latencies = [float(row[3]) for row in rows]
        model_calls = len(rows)
        failures = sum(row[2] != "success" for row in rows)
        return DashboardMetrics(
            model_calls=model_calls,
            average_latency_ms=(sum(latencies) / model_calls) if model_calls else 0.0,
            p95_latency_ms=_percentile(latencies, 0.95),
            error_rate=(failures / model_calls * 100) if model_calls else 0.0,
            rag_calls=sum(int(row[6]) > 0 for row in rows),
            input_tokens=sum(int(row[4] or 0) for row in rows),
            output_tokens=sum(int(row[5] or 0) for row in rows),
            tool_calls=len(tool_rows),
            tool_successes=sum(row[0] == "success" for row in tool_rows),
            tool_failures=sum(row[0] != "success" for row in tool_rows),
            daily_calls=_daily_calls(rows),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_interactions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL DEFAULT 'model',
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    rag_chunk_count INTEGER NOT NULL,
                    rag_enabled INTEGER NOT NULL,
                    tools_enabled INTEGER NOT NULL,
                    memory_enabled INTEGER NOT NULL,
                    error_category TEXT
                );
                CREATE TABLE IF NOT EXISTS tool_interactions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL DEFAULT 'tool',
                    model_interaction_id INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    FOREIGN KEY(model_interaction_id) REFERENCES model_interactions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_model_interactions_timestamp
                    ON model_interactions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_tool_interactions_name
                    ON tool_interactions(tool_name);
                """
            )
            self._ensure_column(
                connection,
                "model_interactions",
                "component",
                "TEXT NOT NULL DEFAULT 'model'",
            )
            self._ensure_column(
                connection,
                "tool_interactions",
                "component",
                "TEXT NOT NULL DEFAULT 'tool'",
            )
            cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
            connection.execute("DELETE FROM tool_interactions WHERE timestamp < ?", (cutoff,))
            connection.execute("DELETE FROM model_interactions WHERE timestamp < ?", (cutoff,))

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        """Upgrade databases created by a previous local development version."""
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _interaction_filter(self, filters: MetricsFilter) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []

        simple_filters = {
            "start_date": ("timestamp >= ?", lambda value: f"{value}T00:00:00+00:00"),
            "end_date": ("timestamp <= ?", lambda value: f"{value}T23:59:59.999999+00:00"),
            "model": ("model = ?", lambda value: value),
            "prompt_version": ("prompt_version = ?", lambda value: value),
            "error_category": ("error_category = ?", lambda value: value),
        }
        for attribute, (clause, transform) in simple_filters.items():
            value = getattr(filters, attribute)
            if value is not None:
                clauses.append(clause)
                parameters.append(transform(value))

        feature_columns = {"rag": "rag_enabled", "tools": "tools_enabled", "memory": "memory_enabled"}
        if column := feature_columns.get(filters.feature):
            clauses.append(f"{column} = 1")
        if filters.tool_name:
            # Qualify the outer id: a bare `id` would resolve to tool_interactions.id.
            clauses.append(
                "EXISTS (SELECT 1 FROM tool_interactions t "
                "WHERE t.model_interaction_id = model_interactions.id AND t.tool_name = ?)"
            )
            parameters.append(filters.tool_name)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else "", parameters)

    @staticmethod
    def _tool_rows(
        connection: sqlite3.Connection,
        interaction_ids: list[int],
        tool_name: str | None,
    ) -> list[sqlite3.Row]:
        if not interaction_ids:
            return []
        placeholders = ", ".join("?" for _ in interaction_ids)
        query = f"SELECT status FROM tool_interactions WHERE model_interaction_id IN ({placeholders})"
        parameters: list[Any] = interaction_ids
        if tool_name:
            query += " AND tool_name = ?"
            parameters.append(tool_name)
        return connection.execute(query, parameters).fetchall()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)


class DisabledMonitoringStore:
    """No-op implementation that avoids creating a database when monitoring is disabled."""

    enabled = False

    def record_model_interaction(self, **_kwargs: Any) -> int:
        return 0

    def record_tool_interaction(self, **_kwargs: Any) -> None:
        return None

    def dashboard(self, _filters: MetricsFilter = MetricsFilter()) -> DashboardMetrics:
        return DashboardMetrics(0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, [])


def create_monitoring_store(config: MonitoringConfig) -> MonitoringStore | DisabledMonitoringStore:
    """Create persistent storage only when monitoring has been explicitly enabled."""
    if not config.enabled:
        return DisabledMonitoringStore()
    return MonitoringStore(config.database_path, config.retention_days)


def estimate_tokens(messages: list[dict[str, Any]] | str) -> int:
    """Estimate token count conservatively when the local provider does not report it."""
    if isinstance(messages, str):
        character_count = len(messages)
    else:
        character_count = sum(len(str(message.get("content", ""))) for message in messages)
    return math.ceil(character_count / 4)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(percentile * len(ordered)) - 1
    return ordered[max(0, index)]


def _daily_calls(rows: list[sqlite3.Row]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        day = str(row[1])[:10]
        counts[day] = counts.get(day, 0) + 1
    return sorted(counts.items())
