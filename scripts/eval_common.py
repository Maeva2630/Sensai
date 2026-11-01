"""Shared harness for the SensAI evaluation scripts (`evaluate_rag.py`, `probe_limits.py`).

The harness drives the real application engine, `web.app.WebApp`, in-process: the same
model runtime, persona prompt, profile, RAG context provider, tool registry and commands
as the terminal and the web page. It only adds two things around it:

- isolation: every run happens in a temporary working directory, so sessions, the RAG
  database, monitoring data and logs never touch the repository or the user's index;
- observation: the chat client is wrapped to record what Ollama reports for each call
  (token counts, durations), without changing the request or the response.

Scoring helpers (text normalisation, keyword matching, abstention detection) live here
too, so both scripts and their unit tests use the same definitions.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from config import AppConfig, load_config  # noqa: E402
from mcp.config import MCPServerConfig  # noqa: E402
from mcp.manager import MCPManager  # noqa: E402
from observability.app_logging import configure_logging  # noqa: E402
from utilities.ollama_client import OllamaError, OllamaHTTPClient  # noqa: E402
from web.app import WebApp  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
REFERENCE_DIR = PROJECT_ROOT / "benchmarks" / "reference"
COURSES_DIR = PROJECT_ROOT / "courses"
DEFAULT_PROFILE = PROJECT_ROOT / "demo" / "profile_camille.json"
TEACHER_MCP_SERVER = SOURCE_DIR / "mcp" / "servers" / "teacher_tools.py"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SETUP_ERROR = 2

# The application sets no output-length limit, and a small model can loop for minutes.
# The harness stops reading a reply after this many streamed chunks (about one token
# each) and flags the turn as a runaway generation instead of waiting.
DEFAULT_MAX_ANSWER_TOKENS = 1500
# A model call whose Ollama load_duration exceeds this was (re)loaded during the turn,
# which explains a latency outlier on a shared Ollama server.
RELOAD_THRESHOLD_MS = 1000


class SetupError(RuntimeError):
    """The evaluation cannot run (Ollama unreachable, model missing, corpus not indexed)."""


# --------------------------------------------------------------------------- scoring

_SPACES = re.compile(r"[\s\u00a0\u202f\u2009\u2007\u2060]+")
_LATEX_FRACTION = re.compile(r"\\[dt]?frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}")
_SPACED_FRACTION = re.compile(r"(\d)\s*/\s*(\d)")
_HOUR = re.compile(r"\b(\d{1,2})(?:\s*h\s*|:)(\d{2})\b")
_DECIMAL_POINT = re.compile(r"(\d)\.(\d)")
_MARKUP = re.compile(r"[*_`#|]+")


def normalize(text: str) -> str:
    """Lower-case, accent-free, single-spaced text for robust keyword matching.

    It also unifies the notations a model may use for the same value: LaTeX fractions
    (``\\frac{3}{4}``), spaced fractions (``3 / 4``), hours (``10 h 30``, ``10:30``),
    decimal points (``0.5`` becomes the French ``0,5``), typographic apostrophes and
    minus signs, and Markdown emphasis.
    """
    if not text:
        return ""
    text = _LATEX_FRACTION.sub(r"\1/\2", text)
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u00ab", '"').replace("\u00bb", '"')
    text = text.replace("\\times", "×").replace("\\cdot", "×")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.lower()
    text = _MARKUP.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _SPACED_FRACTION.sub(r"\1/\2", text)
    text = _HOUR.sub(r"\1h\2", text)
    text = _DECIMAL_POINT.sub(r"\1,\2", text)
    return text.strip()


def _variant_pattern(variant: str) -> re.Pattern[str]:
    """A regex for one accepted variant: `re:` prefix = raw regex, else a bounded literal."""
    if variant.startswith("re:"):
        return re.compile(variant[3:])
    literal = normalize(variant)
    prefix = r"(?<![0-9a-z])" if literal[:1].isalnum() else ""
    suffix = r"(?![0-9a-z])" if literal[-1:].isalnum() else ""
    return re.compile(prefix + re.escape(literal) + suffix)


def matches_variant(normalized_text: str, variant: str) -> bool:
    """Whether an already normalised answer contains one accepted variant."""
    return bool(_variant_pattern(variant).search(normalized_text))


def match_groups(answer: str, groups: Sequence[Sequence[str]]) -> list[bool]:
    """For each group of accepted variants, whether the answer contains at least one."""
    normalized = normalize(answer)
    return [any(matches_variant(normalized, variant) for variant in group) for group in groups]


def is_correct(answer: str, groups: Sequence[Sequence[str]]) -> bool:
    """An answer is correct when every expected group is matched (and there is one)."""
    return bool(groups) and all(match_groups(answer, groups))


# Phrases that mean "the answer is not available" (French and English, normalised form).
ABSTENTION_MARKERS: tuple[str, ...] = (
    "cannot find", "can't find", "could not find", "couldn't find", "not find this information",
    "no information", "not mentioned", "not specified", "not provided", "not available",
    "i don't know", "i do not know", "not in the indexed", "unable to find",
    "je ne trouve pas", "ne trouve pas", "je n'ai pas trouve", "n'ai pas trouve", "introuvable",
    "pas d'information", "aucune information", "pas d'informations", "aucune donnee",
    "pas de donnees", "pas de donnee", "ne dispose pas", "je n'ai pas acces", "n'ai pas acces",
    "pas acces a", "je ne sais pas", "je ne connais pas", "ne connais pas", "pas connaissance",
    "je n'ai pas connaissance", "ne contient pas", "ne contiennent pas", "ne mentionne pas",
    "ne mentionnent pas", "n'est pas mentionne", "n'est pas mentionnee", "ne sont pas mentionne",
    "n'est pas indique", "n'est pas indiquee", "n'est pas precise", "n'est pas precisee",
    "ne precise pas", "ne precisent pas", "ne figure pas", "ne figurent pas", "n'apparait pas",
    "n'est pas fourni", "n'est pas fournie", "non mentionne", "non precise", "non indique",
    "pas mentionne", "pas precise", "pas indique", "pas en mesure", "impossible de",
    "je ne peux pas", "ne peux pas vous", "n'existe pas", "n'existe aucun", "aucun theoreme",
    "pas de theoreme", "aucune trace", "pas de trace", "n'ai aucune", "aucune reference",
    "pas de reference", "ne correspond a aucun", "n'est pas un theoreme", "pas un theoreme",
    "n'y a pas d'exercice", "aucun exercice", "pas d'exercice", "ne comporte pas",
    "ne comporte aucun", "il n'y a pas", "n'y a aucun", "documents indexes", "fictif", "fictive",
    "pas reconnu", "n'est pas reconnu", "ne semble pas exister", "je n'ai pas d'information",
    "n'ai pas d'informations", "manque d'information", "information n'est pas",
    "informations ne sont pas", "pas disponible", "non disponible", "pas d'acces",
)


def is_abstention(answer: str) -> bool:
    """Whether the answer says, in some form, that the information is unavailable."""
    normalized = normalize(answer)
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


# --------------------------------------------------------------------------- statistics

def median(values: Iterable[float | int | None]) -> float | None:
    """Median of the non-missing values, or None when there is none."""
    present = [value for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def rate(numerator: int, denominator: int) -> float | None:
    """A ratio in [0, 1], or None when nothing was measured."""
    return numerator / denominator if denominator else None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def ratio_text(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({percent(rate(numerator, denominator))})"


def fmt_seconds(milliseconds: float | None) -> str:
    return "n/a" if milliseconds is None else f"{milliseconds / 1000:.1f} s"


def fmt_int(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


# --------------------------------------------------------------------------- metadata

def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


def git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
        )
    except OSError:
        return None
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:12]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ollama_metadata(base_url: str) -> dict[str, Any]:
    """Ollama version and the models loaded when the run starts (shared-use evidence)."""
    import requests

    metadata: dict[str, Any] = {"version": None, "loaded_models_at_start": None}
    try:
        metadata["version"] = requests.get(f"{base_url}/api/version", timeout=5).json().get("version")
        loaded = requests.get(f"{base_url}/api/ps", timeout=5).json().get("models", [])
        metadata["loaded_models_at_start"] = [model.get("name") for model in loaded if isinstance(model, dict)]
    except (requests.RequestException, ValueError, AttributeError):
        pass
    return metadata


def machine_metadata() -> dict[str, Any]:
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "python": platform.python_version()}


def check_models(config: AppConfig) -> None:
    """Fail fast, with the command to run, when Ollama or a required model is missing."""
    client = OllamaHTTPClient(base_url=config.model.base_url, model=config.model.chat_model)
    try:
        installed = client.list_models()
    except OllamaError as error:
        raise SetupError(f"Ollama is not reachable at {config.model.base_url}: {error}. Start it with: ollama serve") from error

    def present(name: str) -> bool:
        return any(model == name or model.split(":")[0] == name or model == f"{name}:latest" for model in installed)

    for name in (config.model.chat_model, config.model.embedding_model):
        if not present(name):
            raise SetupError(f"The model '{name}' is not installed. Run: ollama pull {name}")


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


# --------------------------------------------------------------------------- harness

_RAG_SOURCE = re.compile(r"\[Source: (?P<name>[^|\]]+?) \|[^\]]*?Score: (?P<score>-?[0-9.]+)\]")
_SOURCES_EVENT = re.compile(r"([^;:]+?) \(chunks:")
GUARD_PATTERN = re.compile(
    r"context|contexte|window|fenetre|too long|trop long|tokens?\b|tronqu|truncat|limite|depasse|exceed",
    re.IGNORECASE,
)


@dataclass
class TurnRecord:
    """Everything observable about one question sent through the application."""

    question: str
    answer: str = ""
    latency_ms: float | None = None
    first_token_ms: float | None = None
    notices: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tool_lines: list[str] = field(default_factory=list)
    sources_line: str | None = None
    rag_context_used: bool = False
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    answered_by: str | None = None
    # Notices and errors received before any model call of the turn had finished:
    # a warning here was shown before (or instead of) sending the request.
    pre_send_messages: list[str] = field(default_factory=list)
    request_messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int | None:
        counts = [call.get("prompt_eval_count") for call in self.model_calls if call.get("prompt_eval_count") is not None]
        return max(counts) if counts else None

    @property
    def completion_tokens(self) -> int | None:
        counts = [call.get("eval_count") for call in self.model_calls if call.get("eval_count") is not None]
        return sum(counts) if counts else None

    @property
    def runaway(self) -> bool:
        """The reply was cut by the harness because it exceeded the answer-length cap."""
        return any(call.get("stopped_by_harness") for call in self.model_calls)

    @property
    def model_reloaded(self) -> bool:
        return any((call.get("load_duration_ms") or 0) > RELOAD_THRESHOLD_MS for call in self.model_calls)

    @property
    def retrieved_sources(self) -> list[str]:
        return sorted({item["source"] for item in self.retrieved})

    @property
    def tool_names(self) -> list[str]:
        names = []
        for line in self.tool_lines:
            match = re.match(r"\[tool\] ([\w.-]+)\(", line)
            if match:
                names.append(match.group(1))
        return names

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 1),
            "first_token_ms": None if self.first_token_ms is None else round(self.first_token_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "rag_context_used": self.rag_context_used,
            "retrieved": self.retrieved,
            "sources_line": self.sources_line,
            "notices": self.notices,
            "errors": self.errors,
            "tool_calls": self.tool_lines,
            "model_calls": self.model_calls,
            "answered_by": self.answered_by,
            "runaway": self.runaway,
            "model_reloaded": self.model_reloaded,
            "pre_send_messages": self.pre_send_messages,
        }


def parse_retrieved(rag_context: str | None, sources_line: str | None = None) -> list[dict[str, Any]]:
    """Retrieved excerpts (file name and similarity score) from the RAG message or the sources line."""
    retrieved = []
    if rag_context:
        for match in _RAG_SOURCE.finditer(rag_context):
            retrieved.append({"source": match.group("name").strip(), "score": float(match.group("score"))})
    if not retrieved and sources_line:
        body = sources_line.split(":", 1)[1] if ":" in sources_line else sources_line
        for match in _SOURCES_EVENT.finditer(body):
            retrieved.append({"source": match.group(1).strip(), "score": None})
    return retrieved


class AppHarness:
    """One isolated SensAI application instance driven like the web page drives it."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        with_teacher_mcp: bool = False,
        network_enabled: bool = False,
        profile_path: Path = DEFAULT_PROFILE,
        label: str = "eval",
        max_answer_tokens: int = DEFAULT_MAX_ANSWER_TOKENS,
    ) -> None:
        config = load_config(PROJECT_ROOT / "config.json")
        model_config = config.model
        if model:
            model_config = replace(model_config, chat_model=model,
                                   fallback_chat_models=tuple(m for m in model_config.fallback_chat_models if m != model))
        if temperature is not None:
            model_config = replace(model_config, temperature=temperature)
        self.config = replace(config, model=model_config,
                              features=replace(config.features, network_enabled=network_enabled))
        self.with_teacher_mcp = with_teacher_mcp
        self.profile_source = profile_path
        self.label = label
        self.max_answer_tokens = max_answer_tokens
        self.workspace: Path | None = None
        self.app: WebApp | None = None
        self._previous_cwd: str | None = None
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._mcp: MCPManager | None = None
        self._model_calls: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- lifecycle

    def __enter__(self) -> "AppHarness":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def start(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix=f"sensai-{self.label}-")
        self.workspace = Path(self._tmp.name).resolve()
        self._previous_cwd = os.getcwd()
        # sessions/, data/ (RAG and monitoring databases), logs/ and exports/ are relative
        # to the working directory: inside the temporary workspace, nothing reaches the repo.
        os.chdir(self.workspace)
        profile = self.workspace / "profile.json"
        shutil.copyfile(self.profile_source, profile)
        configure_logging(replace(self.config.logging, file_path=str(self.workspace / "logs" / "sensai.log")))
        servers = []
        if self.with_teacher_mcp:
            servers.append(MCPServerConfig(name="teacher", command=sys.executable,
                                           args=[str(TEACHER_MCP_SERVER)], timeout=10))
        self._mcp = MCPManager(servers)
        self.app = WebApp(self.config, profile_path=str(profile), mcp_manager=self._mcp)
        database = Path(self.app.runtime.rag_indexer.vector_store.database_path).resolve()
        if self.workspace not in database.parents:
            raise SetupError("Isolation check failed: the RAG database is outside the temporary workspace.")
        self._instrument(self.app.runtime.chat_client)

    def close(self) -> None:
        if self._mcp is not None:
            self._mcp.close()
            self._mcp = None
        if self._previous_cwd is not None:
            os.chdir(self._previous_cwd)
            self._previous_cwd = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def _instrument(self, client: Any) -> None:
        """Record Ollama's counters for every model call, passing every line through unchanged."""
        calls = self._model_calls
        cap = self.max_answer_tokens
        original_stream = client.stream_message
        original_complete = client.complete_message

        def stream_message(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **options: Any):
            record: dict[str, Any] = {
                "kind": "stream",
                "message_count": len(messages),
                "request_characters": sum(len(str(message.get("content", ""))) for message in messages),
                "tools_offered": [tool.get("function", {}).get("name") for tool in tools or []],
            }
            calls.append(record)
            chunks = 0
            stream = original_stream(messages, tools=tools, **options)
            try:
                for line in stream:
                    if line:
                        _capture_counters(record, line)
                        chunks += 1
                    yield line
                    if cap and chunks > cap and "eval_count" not in record:
                        # Closing the stream closes the HTTP response: Ollama stops generating.
                        record["stopped_by_harness"] = True
                        record["streamed_chunks"] = chunks
                        return
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

        def complete_message(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                             output_format: Any = None):
            record: dict[str, Any] = {
                "kind": "complete",
                "message_count": len(messages),
                "request_characters": sum(len(str(message.get("content", ""))) for message in messages),
                "structured": output_format is not None,
            }
            calls.append(record)
            response = original_complete(messages, tools=tools, output_format=output_format)
            try:
                _capture_counters(record, response.content)
            except AttributeError:
                pass
            return response

        client.stream_message = stream_message
        client.complete_message = complete_message

    # ---------------------------------------------------------------- actions

    @property
    def model_calls(self) -> list[dict[str, Any]]:
        return self._model_calls

    def command(self, text: str) -> str:
        assert self.app is not None
        return self.app.run_command(text)

    def set_rag(self, enabled: bool) -> None:
        output = self.command("/rag on" if enabled else "/rag off")
        if self.app.session.rag_enabled != enabled:
            raise SetupError(f"Could not switch RAG {'on' if enabled else 'off'}: {output}")

    def ingest(self, paths: Sequence[Path]) -> str:
        """Index documents through the application's own `/ingest --multiple` command."""
        quoted = " ".join(f'"{path}"' for path in paths)
        output = self.command(f"/ingest --multiple {quoted}")
        expected = f"{len(paths)}/{len(paths)} paths indexed"
        if expected not in output:
            raise SetupError(f"Indexing failed: {output.strip()}")
        return output

    def indexed_documents(self) -> list[tuple[str, int]]:
        assert self.app is not None
        return self.app.runtime.rag_indexer.vector_store.list_sources()

    def ask(self, question: str, *, fresh: bool = True) -> TurnRecord:
        """Send one message as the web page does and collect every event it produces.

        `fresh` starts a new conversation first, so answers never leak between questions.
        A confirmation request (web search, network tool) is refused, like a teacher
        clicking Cancel: nothing leaves the machine during an evaluation.
        """
        assert self.app is not None
        if fresh:
            self.app.new_conversation()
        record = TurnRecord(question=question)
        first_call = len(self._model_calls)
        started = perf_counter()
        parts: list[str] = []
        for event in self.app.stream_chat(question):
            kind = event.get("type")
            if kind in ("notice", "error") and not self._calls_finished(first_call):
                record.pre_send_messages.append(event.get("text", ""))
            if kind == "token":
                if record.first_token_ms is None:
                    record.first_token_ms = (perf_counter() - started) * 1000
                parts.append(event.get("text", ""))
            elif kind == "confirm":
                self.app.resolve_confirmation(event["id"], False)
                record.notices.append(f"confirmation refused: {event.get('question', '')}")
            elif kind == "notice":
                record.notices.append(event.get("text", ""))
            elif kind == "error":
                record.errors.append(event.get("text", ""))
            elif kind == "tool":
                record.tool_lines.append(event.get("text", ""))
            elif kind == "sources" and str(event.get("text", "")).startswith("Sources:"):
                record.sources_line = event.get("text")
        record.latency_ms = (perf_counter() - started) * 1000
        record.answer = "".join(parts).strip()
        # The retrieved excerpts travel in the user turn of the request, framed as data
        # (`last_rag_context` only holds SensAI's RAG rules).
        last_request = getattr(self.app.session, "last_request_messages", None) or []
        excerpts = next(
            (str(message.get("content", "")) for message in last_request
             if message.get("role") == "user" and str(message.get("content", "")).startswith("<<<DOCUMENT EXCERPTS>>>")),
            None,
        )
        record.rag_context_used = excerpts is not None
        record.retrieved = parse_retrieved(excerpts, record.sources_line)
        record.model_calls = [dict(call) for call in self._model_calls[first_call:]]
        record.answered_by = self.app._active_model()
        record.request_messages = [dict(message) for message in last_request]
        return record

    def _calls_finished(self, first_call: int) -> int:
        return sum(1 for call in self._model_calls[first_call:]
                   if "eval_count" in call or call.get("stopped_by_harness"))

    @staticmethod
    def guard_messages(record: TurnRecord, before_send_only: bool = True) -> list[str]:
        """Notices or errors about the size of the request (a context guard).

        With `before_send_only`, only messages shown before any model call finished count:
        the existing post-answer budget warning does not prevent a truncated request.
        """
        texts = record.pre_send_messages if before_send_only else [*record.notices, *record.errors]
        return [text for text in texts if GUARD_PATTERN.search(normalize(text))]

    def describe(self) -> dict[str, Any]:
        model, rag = self.config.model, self.config.rag
        return {
            "pipeline": "web.app.WebApp in-process (same engine as the CLI and the web page)",
            "chat_model": model.chat_model,
            "fallback_chat_models": list(model.fallback_chat_models),
            "embedding_model": model.embedding_model,
            "temperature": model.temperature,
            "context_window": model.context_window,
            "rag_top_k": rag.top_k,
            "rag_relevance_threshold": rag.relevance_threshold,
            "rag_max_context_characters": rag.max_context_characters,
            "persona": self.config.personas.active.identifier,
            "profile": relative(self.profile_source),
            "mcp_servers": ["teacher (local stdio)"] if self.with_teacher_mcp else [],
            "network_tools": self.config.features.network_enabled,
            "max_answer_tokens": self.max_answer_tokens or None,
            "workspace": "temporary directory, deleted after the run",
        }


def _capture_counters(record: dict[str, Any], raw_line: bytes | str) -> None:
    """Copy Ollama's final counters (tokens, durations) from one NDJSON line into a record."""
    try:
        payload = json.loads(raw_line)
    except (ValueError, TypeError):
        return
    if not isinstance(payload, dict) or not payload.get("done"):
        return
    for key in ("prompt_eval_count", "eval_count", "done_reason"):
        if key in payload:
            record[key] = payload[key]
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            record[f"{key}_ms"] = round(value / 1e6, 1)
