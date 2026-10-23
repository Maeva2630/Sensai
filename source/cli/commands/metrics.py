"""Local monitoring dashboard command."""

import shlex
import sqlite3
from datetime import date

from cli.session import CliCommandContext, CliSession, CommandDefinition
from observability.monitoring import DashboardMetrics, MetricsFilter


def _dispatch_metrics(
    user_input: str,
    _session: CliSession,
    context: CliCommandContext,
) -> None:
    """Render local interaction metrics without contacting Ollama."""
    if context.monitoring_store is None or not context.monitoring_store.enabled:
        print("Monitoring is disabled in config.json.")
        return
    try:
        filters = _parse_filters(user_input)
    except ValueError as error:
        print(f"Invalid /metrics arguments: {error}")
        _print_usage()
        return
    try:
        dashboard = context.monitoring_store.dashboard(filters)
    except (OSError, sqlite3.Error) as error:
        print(f"Monitoring dashboard is unavailable: {error}")
        return
    print(_format_dashboard(dashboard, filters))


def _parse_filters(user_input: str) -> MetricsFilter:
    tokens = shlex.split(user_input)
    if not tokens or tokens[0] != "/metrics":
        raise ValueError("the command must start with /metrics")
    options = {
        "--from": "start_date",
        "--to": "end_date",
        "--model": "model",
        "--prompt-version": "prompt_version",
        "--feature": "feature",
        "--tool": "tool_name",
        "--error": "error_category",
    }
    values: dict[str, str] = {}
    for index in range(1, len(tokens), 2):
        if tokens[index] not in options or index + 1 >= len(tokens):
            raise ValueError(f"unknown or incomplete option: {tokens[index]}")
        values[options[tokens[index]]] = tokens[index + 1]
    if values.get("feature") and values["feature"] not in {"rag", "tools", "memory"}:
        raise ValueError("--feature must be rag, tools, or memory")
    for option in ("start_date", "end_date"):
        if option in values:
            try:
                date.fromisoformat(values[option])
            except ValueError as error:
                raise ValueError(f"{option.replace('_', ' ')} must use YYYY-MM-DD") from error
    if values.get("start_date") and values.get("end_date") and values["start_date"] > values["end_date"]:
        raise ValueError("--from must not be later than --to")
    return MetricsFilter(**values)


def _format_dashboard(metrics: DashboardMetrics, filters: MetricsFilter) -> str:
    filter_values = [
        f"{name}={value}"
        for name, value in vars(filters).items()
        if value is not None
    ]
    lines = ["Local monitoring dashboard"]
    if filter_values:
        lines.append(f"Filters: {', '.join(filter_values)}")
    lines.extend(
        [
            f"Model calls: {metrics.model_calls}",
            f"Latency: average {metrics.average_latency_ms:.1f} ms | p95 {metrics.p95_latency_ms:.1f} ms",
            f"Error rate: {metrics.error_rate:.1f}%",
            f"RAG retrievals: {metrics.rag_calls}",
            f"Estimated tokens: input {metrics.input_tokens} | output {metrics.output_tokens}",
            (
                "Tool calls: "
                f"{metrics.tool_calls} ({metrics.tool_successes} succeeded, {metrics.tool_failures} failed)"
            ),
            "Calls over time:",
        ]
    )
    lines.extend(f"  {day}: {count}" for day, count in metrics.daily_calls)
    if not metrics.daily_calls:
        lines.append("  No recorded interactions.")
    return "\n".join(lines)


def _print_usage() -> None:
    print("Usage: /metrics [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--model NAME]")
    print("                [--prompt-version VERSION]")
    print("                [--feature rag|tools|memory] [--tool NAME] [--error CATEGORY]")


METRICS_COMMANDS: tuple[CommandDefinition, ...] = (
    CommandDefinition(
        name="/metrics",
        usage="/metrics [--from DATE] [--to DATE] [--model NAME] [--prompt-version VERSION] [--feature NAME] [--tool NAME] [--error CATEGORY]",
        description="Display local call, latency, error, token, RAG, and tool metrics.",
        feature="Observability",
        handler=_dispatch_metrics,
    ),
)
