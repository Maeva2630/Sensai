#!/usr/bin/env python3
"""Run reproducible local Ollama benchmarks for the SensAI teacher scenario.

The script never downloads, modifies, or selects models. It sends the fixed prompt set
to a locally running Ollama server, records API timing/token metrics, and samples local
RAM plus NVIDIA VRAM when `nvidia-smi` is available.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPTS = Path("benchmarks/prompts.json")
DEFAULT_RESULTS_DIR = Path("benchmarks/results")
POLL_INTERVAL_SECONDS = 0.25


def _command_output(*command: str) -> str | None:
    """Return a command's output, or None when it cannot run successfully."""
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _cpu_model() -> str | None:
    """Read the CPU model without relying on the locale of `lscpu`."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return None


def _mem_available_mib() -> float | None:
    """Read currently available Linux memory in MiB."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _mem_total_mib() -> float | None:
    """Read total Linux memory in MiB."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _ollama_rss_mib() -> float | None:
    """Return summed RSS for local Ollama server/runner processes in MiB."""
    rss_kib = 0
    found = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
            if "ollama" not in command.lower():
                continue
            status = (entry / "status").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                rss_kib += int(line.split()[1])
                found = True
                break
    return rss_kib / 1024 if found else None


def _nvidia_stats() -> dict[str, float] | None:
    """Read NVIDIA GPU memory and utilisation in MiB/percent, when available."""
    output = _command_output(
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return None
    try:
        used, total, utilization = (float(value.strip()) for value in output.splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"used_mib": used, "total_mib": total, "utilization_percent": utilization}


def _nvidia_device_info() -> dict[str, str | float] | None:
    """Read stable NVIDIA device information for the benchmark environment."""
    output = _command_output(
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return None
    try:
        name, memory_total, driver_version = (value.strip() for value in output.splitlines()[0].split(","))
        return {"name": name, "memory_total_mib": float(memory_total), "driver_version": driver_version}
    except (ValueError, IndexError):
        return None


class ResourceSampler:
    """Periodically sample system resources while one model request runs."""

    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._append_sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict[str, float | int | None]:
        """Return peaks and deltas; values remain measurements, not model-size estimates."""
        if not self.samples:
            return {"sample_count": 0}
        available = [sample["ram_available_mib"] for sample in self.samples if sample["ram_available_mib"] is not None]
        rss = [sample["ollama_rss_mib"] for sample in self.samples if sample["ollama_rss_mib"] is not None]
        gpu = [sample["gpu_used_mib"] for sample in self.samples if sample["gpu_used_mib"] is not None]
        gpu_utilization = [sample["gpu_utilization_percent"] for sample in self.samples if sample["gpu_utilization_percent"] is not None]
        return {
            "sample_count": len(self.samples),
            "ram_available_start_mib": available[0] if available else None,
            "ram_available_min_mib": min(available) if available else None,
            "ram_available_drop_mib": available[0] - min(available) if available else None,
            "ollama_rss_peak_mib": max(rss) if rss else None,
            "gpu_memory_start_mib": gpu[0] if gpu else None,
            "gpu_memory_peak_mib": max(gpu) if gpu else None,
            "gpu_memory_delta_mib": max(gpu) - gpu[0] if gpu else None,
            "gpu_utilization_peak_percent": max(gpu_utilization) if gpu_utilization else None,
        }

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(POLL_INTERVAL_SECONDS)
            if not self._stop.is_set():
                self._append_sample()

    def _append_sample(self) -> None:
        gpu = _nvidia_stats()
        self.samples.append(
            {
                "time_monotonic": time.monotonic(),
                "ram_available_mib": _mem_available_mib(),
                "ollama_rss_mib": _ollama_rss_mib(),
                "gpu_used_mib": gpu["used_mib"] if gpu else None,
                "gpu_utilization_percent": gpu["utilization_percent"] if gpu else None,
            }
        )


def _request_json(url: str, payload: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    """Post a streaming Ollama request and return receipt times plus NDJSON chunks."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=300) as response:  # noqa: S310 -- local Ollama URL is an explicit argument
            chunks: list[tuple[float, dict[str, Any]]] = []
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    chunks.append((time.perf_counter(), json.loads(line)))
            return chunks
    except HTTPError as error:
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}") from error
    except URLError as error:
        raise RuntimeError(f"Could not contact Ollama: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON while streaming a response.") from error


def _run_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Run one fixed case and record API data, first-token latency, and resource peaks."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "tools": case.get("tools", []),
        "options": options,
        "stream": True,
        "keep_alive": "10m",
    }
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()
    try:
        chunks = _request_json(f"{base_url.rstrip('/')}/api/chat", payload)
    finally:
        sampler.stop()
    finished = time.perf_counter()

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    first_text_chunk_index: int | None = None
    first_event_chunk_index: int | None = None
    final_chunk: dict[str, Any] = {}
    first_text_latency_ms: float | None = None
    first_event_latency_ms: float | None = None
    for index, (received_at, chunk) in enumerate(chunks):
        message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
        content = message.get("content", "")
        calls = message.get("tool_calls", [])
        if content:
            text_parts.append(content)
            if first_text_chunk_index is None:
                first_text_chunk_index = index
                first_text_latency_ms = (received_at - started) * 1000
        if calls:
            tool_calls.extend(calls)
        if (content or calls) and first_event_chunk_index is None:
            first_event_chunk_index = index
            first_event_latency_ms = (received_at - started) * 1000
        if chunk.get("done") is True:
            final_chunk = chunk

    return {
        "case_id": case["id"],
        "purpose": case["purpose"],
        "status": "success",
        "wall_total_latency_ms": (finished - started) * 1000,
        "first_text_token_latency_ms": first_text_latency_ms,
        "first_response_event_latency_ms": first_event_latency_ms,
        "first_text_chunk_index": first_text_chunk_index,
        "first_response_event_chunk_index": first_event_chunk_index,
        "response_text": "".join(text_parts),
        "tool_calls": tool_calls,
        "tool_call_requested": bool(tool_calls),
        "manual_quality_checks": case.get("manual_quality_checks", []),
        "ollama": _ollama_metrics(final_chunk),
        "resources": sampler.summary(),
        "measurement_note": "Wall and first-token latencies plus resource values are measured locally. Ollama duration/token fields are provider-reported.",
    }


def _ollama_metrics(chunk: dict[str, Any]) -> dict[str, float | int | None]:
    """Convert official Ollama nanosecond fields to milliseconds and token rates."""
    def nanoseconds_to_ms(name: str) -> float | None:
        value = chunk.get(name)
        return value / 1_000_000 if isinstance(value, int) else None

    eval_count = chunk.get("eval_count") if isinstance(chunk.get("eval_count"), int) else None
    eval_duration = chunk.get("eval_duration") if isinstance(chunk.get("eval_duration"), int) else None
    prompt_count = chunk.get("prompt_eval_count") if isinstance(chunk.get("prompt_eval_count"), int) else None
    prompt_duration = chunk.get("prompt_eval_duration") if isinstance(chunk.get("prompt_eval_duration"), int) else None
    return {
        "load_duration_ms": nanoseconds_to_ms("load_duration"),
        "total_duration_ms": nanoseconds_to_ms("total_duration"),
        "prompt_eval_count": prompt_count,
        "prompt_eval_duration_ms": nanoseconds_to_ms("prompt_eval_duration"),
        "eval_count": eval_count,
        "eval_duration_ms": nanoseconds_to_ms("eval_duration"),
        "output_tokens_per_second": (eval_count * 1_000_000_000 / eval_duration) if eval_count and eval_duration else None,
        "input_tokens_per_second": (prompt_count * 1_000_000_000 / prompt_duration) if prompt_count and prompt_duration else None,
    }


def _environment() -> dict[str, Any]:
    """Collect reproducible machine metadata without guessing unavailable values."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "operating_system": platform.platform(),
        "cpu_model": _cpu_model(),
        "ram_total_mib": _mem_total_mib(),
        "ram_available_mib_at_start": _mem_available_mib(),
        "nvidia_gpu": _nvidia_device_info(),
        "nvidia_gpu_at_start": _nvidia_stats(),
        "ollama_version": _command_output("ollama", "--version"),
    }


def _load_prompts(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read prompt set '{path}': {error}") from error
    if not isinstance(data.get("cases"), list) or not isinstance(data.get("protocol"), dict):
        raise SystemExit("Prompt set must contain object fields named 'protocol' and 'cases'.")
    return data


def _write_results(results_dir: Path, model: str, result: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "-").replace("/", "-")
    path = results_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_model}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _print_summary(result: dict[str, Any]) -> None:
    print(f"Model: {result['model']}")
    print(f"Prompt set: {result['prompt_set_version']}")
    for run in result["runs"]:
        metrics = run["ollama"]
        resources = run["resources"]
        print(
            f"- {run['case_id']}: total={run['wall_total_latency_ms']:.1f} ms, "
            f"output={metrics['output_tokens_per_second'] or 0:.2f} tok/s, "
            f"GPU peak={resources.get('gpu_memory_peak_mib')} MiB, "
            f"Ollama RSS peak={resources.get('ollama_rss_peak_mib')} MiB, "
            f"tool_call={run['tool_call_requested']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Installed Ollama model name, for example llama3.2:latest")
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions of every prompt case (default: 3)")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS, help="Prompt-set JSON file")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for generated JSON results")
    parser.add_argument("--cold-run", action="store_true", help="Label the first run as cold after you manually ran 'ollama stop <model>'")
    arguments = parser.parse_args()
    if arguments.runs < 1:
        raise SystemExit("--runs must be at least 1")

    prompt_set = _load_prompts(arguments.prompts)
    result: dict[str, Any] = {
        "model": arguments.model,
        "prompt_set_version": prompt_set["version"],
        "protocol": prompt_set["protocol"],
        "environment": _environment(),
        "runs": [],
    }
    for repetition in range(arguments.runs):
        for case_index, case in enumerate(prompt_set["cases"]):
            print(f"Running {arguments.model}: {case['id']} ({repetition + 1}/{arguments.runs})...", flush=True)
            run = _run_case(arguments.base_url, arguments.model, case, prompt_set["protocol"])
            run["repetition"] = repetition + 1
            run["cold_run"] = arguments.cold_run and repetition == 0 and case_index == 0
            result["runs"].append(run)

    path = _write_results(arguments.results_dir, arguments.model, result)
    _print_summary(result)
    print(f"Raw measured results written to {path}")


if __name__ == "__main__":
    main()
