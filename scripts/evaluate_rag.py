#!/usr/bin/env python3
"""A/B evaluation of SensAI's RAG mode: the same questions with RAG off, then on.

The questions come from a business evaluation set written for Camille, a 6e mathematics
teacher (`benchmarks/eval/rag_eval_fr.json`). The corpus is the fictional course folder
`courses/`. Each question is sent through the real application engine (`web.app.WebApp`),
in a fresh conversation, with RAG switched by the application's own `/rag on|off`
command. Nothing is written to the repository except the result file.

Metrics per condition: accuracy on answerable questions (keyword scoring), correct
abstention and hallucination on questions whose answer is absent from the corpus, median
latency and Ollama token counts. The script prints a table and a verdict, saves every
answer as JSON under `benchmarks/results/`, and exits with 0 (PASS) or 1 (FAIL).

    python3 scripts/evaluate_rag.py --quick          # about one minute, live demo
    python3 scripts/evaluate_rag.py                  # full set, 3 runs
    python3 scripts/evaluate_rag.py --model llama3.2:latest --runs 5
    python3 scripts/evaluate_rag.py --rescore benchmarks/reference/rag_eval_qwen2.5-3b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Sequence

import eval_common as common

DEFAULT_EVAL_SET = common.PROJECT_ROOT / "benchmarks" / "eval" / "rag_eval_fr.json"
DEFAULT_RUNS = 3
DEFAULT_MIN_GAIN_POINTS = 30.0
DEFAULT_MIN_ABSTENTION = 0.75
CONDITIONS = ("off", "on")


# --------------------------------------------------------------------------- data

def load_eval_set(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise common.SetupError(f"{path.name} has no questions.")
    for question in questions:
        if question.get("type") not in {"answerable", "absent", "control"}:
            raise common.SetupError(f"Question {question.get('id')} has an unknown type.")
        if question["type"] != "absent" and not question.get("expected"):
            raise common.SetupError(f"Question {question['id']} needs expected keywords.")
    return data


def select_questions(eval_set: dict[str, Any], quick: bool) -> list[dict[str, Any]]:
    """All questions, or the `quick` subset (no control questions) for a live demo."""
    questions = eval_set["questions"]
    if quick:
        return [q for q in questions if q.get("quick") and q["type"] != "control"]
    return list(questions)


def corpus_paths(eval_set: dict[str, Any]) -> list[Path]:
    paths = [common.PROJECT_ROOT / entry for entry in eval_set.get("corpus", [])]
    missing = [common.relative(path) for path in paths if not path.is_file()]
    if missing or not paths:
        raise common.SetupError(f"Corpus files are missing: {', '.join(missing) or 'none listed'}")
    return paths


# --------------------------------------------------------------------------- scoring

def score_answer(question: dict[str, Any], answer: str, runaway: bool = False) -> dict[str, Any]:
    """Score one answer against its question type.

    answerable / control: `correct` when every expected keyword group is present;
    otherwise the answer `abstained` (said it does not know) or is `wrong`.
    absent: `abstained` is the expected behaviour; an answer that states a concrete
    unsupported value (a question's `invented` patterns) is `hallucinated`; anything
    else is `evasive`. An abstention that still contains such a value is flagged
    `hedged_guess` for manual review.
    A reply cut by the harness length cap is `runaway`, never correct.
    """
    abstained = common.is_abstention(answer)
    normalized = common.normalize(answer)
    if question["type"] == "absent":
        invented = any(common.matches_variant(normalized, pattern) for pattern in question.get("invented", []))
        if runaway:
            outcome = "runaway"
        elif abstained:
            outcome = "abstained"
        elif invented:
            outcome = "hallucinated"
        else:
            outcome = "evasive"
        return {"outcome": outcome, "correct": None, "abstained": outcome == "abstained",
                "hallucinated": outcome == "hallucinated", "hedged_guess": outcome == "abstained" and invented}
    groups = common.match_groups(answer, question["expected"])
    correct = bool(groups) and all(groups) and not runaway
    if runaway:
        outcome = "runaway"
    elif correct:
        outcome = "correct"
    else:
        outcome = "abstained" if abstained else "wrong"
    return {"outcome": outcome, "correct": correct, "abstained": outcome == "abstained",
            "hallucinated": None, "groups_matched": groups}


def summarize(results: Sequence[dict[str, Any]], runs: int) -> dict[str, Any]:
    """Aggregate the scored answers of each condition."""
    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        answerable = [row for row in rows if row["type"] == "answerable"]
        absent = [row for row in rows if row["type"] == "absent"]
        control = [row for row in rows if row["type"] == "control"]
        with_source = [row for row in answerable if row.get("source")]
        per_run = []
        for run in range(1, runs + 1):
            run_rows = [row for row in answerable if row["run"] == run]
            per_run.append(common.rate(sum(row["correct"] for row in run_rows), len(run_rows)))
        summary[condition] = {
            "answerable_total": len(answerable),
            "answerable_correct": sum(row["correct"] for row in answerable),
            "answerable_abstained": sum(row["outcome"] == "abstained" for row in answerable),
            "answerable_wrong": sum(row["outcome"] == "wrong" for row in answerable),
            "accuracy": common.rate(sum(row["correct"] for row in answerable), len(answerable)),
            "accuracy_per_run": per_run,
            "absent_total": len(absent),
            "absent_abstained": sum(row["abstained"] for row in absent),
            "absent_hallucinated": sum(row["hallucinated"] for row in absent),
            "absent_evasive": sum(row["outcome"] == "evasive" for row in absent),
            "absent_hedged_guess": sum(bool(row.get("hedged_guess")) for row in absent),
            "abstention_rate": common.rate(sum(row["abstained"] for row in absent), len(absent)),
            "control_total": len(control),
            "control_correct": sum(row["correct"] for row in control),
            "rag_context_used": sum(row["rag_context_used"] for row in rows),
            "rag_skipped": sum(not row["rag_context_used"] for row in rows) if condition == "on" else None,
            "retrieval_hits": sum(row["source"] in row["retrieved_sources"] for row in with_source),
            "retrieval_total": len(with_source) if condition == "on" else 0,
            "tool_calls": sum(len(row["tool_calls"]) for row in rows),
            "runaway": sum(row["outcome"] == "runaway" for row in rows),
            "model_reloads": sum(bool(row.get("model_reloaded")) for row in rows),
            "median_latency_ms": common.median(row["latency_ms"] for row in rows),
            "median_first_token_ms": common.median(row["first_token_ms"] for row in rows),
            "median_prompt_tokens": common.median(row["prompt_tokens"] for row in rows),
            "median_completion_tokens": common.median(row["completion_tokens"] for row in rows),
            "turns": len(rows),
        }
    return summary


def verdict(summary: dict[str, Any], min_gain_points: float, min_abstention: float) -> dict[str, Any]:
    """The two pass criteria: accuracy gain from RAG, and abstention on absent facts with RAG on."""
    off, on = summary["off"], summary["on"]
    checks = []
    if off["accuracy"] is None or on["accuracy"] is None:
        checks.append({"name": "accuracy_gain", "passed": False, "line": "RAG accuracy: no answerable question was run  FAIL"})
    else:
        # Rounded so that a gain of exactly the threshold is not lost to floating-point error.
        gain = round((on["accuracy"] - off["accuracy"]) * 100, 6)
        passed = gain >= min_gain_points
        checks.append({
            "name": "accuracy_gain", "passed": passed, "gain_points": round(gain, 1),
            "threshold_points": min_gain_points,
            "line": (f"RAG accuracy: off {common.percent(off['accuracy'])} → on {common.percent(on['accuracy'])} "
                     f"({gain:+.0f} pts, threshold +{min_gain_points:.0f})  {'PASS' if passed else 'FAIL'}"),
        })
    if on["absent_total"]:
        abstention = on["abstention_rate"]
        passed = abstention >= min_abstention
        checks.append({
            "name": "abstention_rag_on", "passed": passed, "rate": abstention, "threshold": min_abstention,
            "line": (f"Abstention on absent facts (RAG on): {on['absent_abstained']}/{on['absent_total']} "
                     f"({common.percent(abstention)}, threshold {min_abstention * 100:.0f}%)  {'PASS' if passed else 'FAIL'}"),
        })
    passed = all(check["passed"] for check in checks)
    return {"passed": passed, "checks": checks, "line": f"VERDICT: {'PASS' if passed else 'FAIL'}"}


# --------------------------------------------------------------------------- report

def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def format_report(summary: dict[str, Any], results: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]],
                  result_verdict: dict[str, Any], runs: int) -> str:
    off, on = summary["off"], summary["on"]
    lines = ["", "Results", "-------"]
    header = f"{'Condition':<9} {'Answerable correct':<20} {'Absent: abstained':<19} {'Hallucinated':<13} {'Control':<9} {'Median latency':<15} {'Median tokens in/out'}"
    lines.append(header)
    for name, data in (("RAG off", off), ("RAG on", on)):
        control = f"{data['control_correct']}/{data['control_total']}" if data["control_total"] else "-"
        abstained = f"{data['absent_abstained']}/{data['absent_total']}"
        hallucinated = f"{data['absent_hallucinated']}/{data['absent_total']}"
        tokens = f"{common.fmt_int(data['median_prompt_tokens'])}/{common.fmt_int(data['median_completion_tokens'])}"
        lines.append(
            f"{name:<9} {common.ratio_text(data['answerable_correct'], data['answerable_total']):<20} "
            f"{abstained:<19} {hallucinated:<13} {control:<9} {common.fmt_seconds(data['median_latency_ms']):<15} {tokens}"
        )
    lines += ["", f"Per question (correct or abstained runs out of {runs}, off → on):"]
    for question in questions:
        rows = [row for row in results if row["id"] == question["id"]]
        cells = []
        for condition in CONDITIONS:
            condition_rows = [row for row in rows if row["condition"] == condition]
            good = sum(row["abstained"] if question["type"] == "absent" else row["correct"] for row in condition_rows)
            cells.append(f"{good}/{len(condition_rows)}")
        on_rows = [row for row in rows if row["condition"] == "on"]
        notes = []
        if question.get("source"):
            hits = sum(question["source"] in row["retrieved_sources"] for row in on_rows)
            notes.append(f"{question['source']} retrieved {hits}/{len(on_rows)}")
        skipped = sum(not row["rag_context_used"] for row in on_rows)
        if skipped:
            notes.append(f"RAG skipped {skipped}/{len(on_rows)} (request claimed by a tool)")
        tools = sorted({name for row in rows for name in row["tool_names"]})
        if tools:
            notes.append("tools: " + ", ".join(tools))
        kind = {"answerable": "", "absent": " [absent]", "control": " [control]"}[question["type"]]
        lines.append(f"  {question['id']} {_short(question['question'], 58):<58}{kind:<10} {cells[0]:>5} → {cells[1]:<5} {'; '.join(notes)}")

    lines += ["", "Diagnostics (not part of the verdict)"]
    if on["retrieval_total"]:
        lines.append(f"  Expected document among retrieved excerpts (RAG on): {common.ratio_text(on['retrieval_hits'], on['retrieval_total'])}")
    if on["rag_skipped"]:
        lines.append(f"  RAG on but no excerpt sent (a tool claimed the request): {on['rag_skipped']}/{on['turns']} turns")
    for name, data in (("off", off), ("on", on)):
        extra = []
        if data["absent_evasive"]:
            extra.append(f"evasive {data['absent_evasive']}")
        if data["absent_hedged_guess"]:
            extra.append(f"abstained but added a guess {data['absent_hedged_guess']}")
        if data["runaway"]:
            extra.append(f"runaway replies cut at the length cap {data['runaway']}")
        if data["model_reloads"]:
            extra.append(f"turns with a model (re)load {data['model_reloads']}")
        if extra:
            lines.append(f"  RAG {name}: " + "; ".join(extra))
    for name, data in (("off", off), ("on", on)):
        per_run = ", ".join(common.percent(value) for value in data["accuracy_per_run"])
        lines.append(f"  Accuracy per run, RAG {name}: {per_run}; wrong answers {data['answerable_wrong']}, "
                     f"abstentions {data['answerable_abstained']} (answerable questions)")
    if off["control_total"]:
        lines.append(f"  Control questions (general knowledge, not in the corpus): off {off['control_correct']}/{off['control_total']}, "
                     f"on {on['control_correct']}/{on['control_total']}")
    lines += ["", "Verdict", "-------"]
    lines += [check["line"] for check in result_verdict["checks"]]
    lines.append(result_verdict["line"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- run

def run_evaluation(
    harness: common.AppHarness,
    questions: Sequence[dict[str, Any]],
    runs: int,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Ask every question in both conditions, `runs` times, each time in a new conversation.

    The order of the two conditions alternates between runs so that neither condition
    always benefits from a model that has just been loaded.
    """
    results = []
    total = len(questions) * len(CONDITIONS) * runs
    done = 0
    for run in range(1, runs + 1):
        order = CONDITIONS if run % 2 else tuple(reversed(CONDITIONS))
        for question in questions:
            for condition in order:
                harness.set_rag(condition == "on")
                record = harness.ask(question["question"])
                score = score_answer(question, record.answer, record.runaway)
                done += 1
                row = {
                    "run": run, "condition": condition, "id": question["id"], "type": question["type"],
                    "source": question.get("source"), "tags": question.get("tags", []),
                    **score, **record.as_dict(),
                    "retrieved_sources": record.retrieved_sources, "tool_names": record.tool_names,
                }
                results.append(row)
                progress(f"  [{done:>3}/{total}] run {run} {question['id']} RAG {condition:<3} {score['outcome']:<12} "
                         f"{common.fmt_seconds(record.latency_ms)}")
    return results


def rescore(path: Path, eval_set_path: Path, min_gain: float, min_abstention: float) -> int:
    """Score the answers saved in a result file again with the current evaluation set.

    No model is called: the answers are the recorded ones. This makes the scoring
    auditable (a keyword variant can be corrected and its effect shown) and keeps the
    previous verdict in the file for comparison.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    eval_set = load_eval_set(eval_set_path)
    by_id = {question["id"]: question for question in eval_set["questions"]}
    changed = 0
    for row in data["results"]:
        question = by_id.get(row["id"])
        if question is None:
            raise common.SetupError(f"Question {row['id']} is not in {eval_set_path.name}.")
        before = row["outcome"]
        row.update(score_answer(question, row["answer"], bool(row.get("runaway"))))
        changed += row["outcome"] != before
    asked = [by_id[question_id] for question_id in dict.fromkeys(row["id"] for row in data["results"])]
    summary = summarize(data["results"], data["runs"])
    result_verdict = verdict(summary, min_gain, min_abstention)
    data.setdefault("rescoring", []).append({
        "at": common.utc_timestamp(), "eval_set_sha256_12": common.file_digest(eval_set_path),
        "outcomes_changed": changed, "verdict_before": data["verdict"]["line"],
    })
    data.update({"summary": summary, "verdict": result_verdict, "eval_set_sha256_12": common.file_digest(eval_set_path),
                 "thresholds": {"min_gain_points": min_gain, "min_abstention": min_abstention}})
    print(f"Re-scored {len(data['results'])} saved answers from {common.relative(path)} "
          f"({data['settings']['chat_model']}, recorded {data['created_at']}): {changed} outcome(s) changed.")
    print(format_report(summary, data["results"], asked, result_verdict, data["runs"]))
    common.write_json(path, data)
    return common.EXIT_PASS if result_verdict["passed"] else common.EXIT_FAIL


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B evaluation of SensAI RAG mode (on vs off) with a PASS/FAIL verdict.")
    parser.add_argument("--runs", type=int, default=None, help=f"Repetitions of the whole set (default {DEFAULT_RUNS}; 1 with --quick).")
    parser.add_argument("--quick", action="store_true", help="One run on the quick subset (about one minute).")
    parser.add_argument("--model", help="Chat model to evaluate (default: chat_model in config.json).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default: config.json).")
    parser.add_argument("--persona", help="Persona identifier to activate with /persona use (default: config.json).")
    parser.add_argument("--with-mcp", action="store_true", help="Also start the local teacher MCP server (its tools are offered on every request).")
    parser.add_argument("--max-answer-tokens", type=int, default=common.DEFAULT_MAX_ANSWER_TOKENS,
                        help=f"Stop reading a reply after this many streamed tokens and score it 'runaway' (default {common.DEFAULT_MAX_ANSWER_TOKENS}; 0 = no cap).")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="Evaluation set JSON file.")
    parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN_POINTS, help="Minimum accuracy gain, in points, for PASS (default 30).")
    parser.add_argument("--min-abstention", type=float, default=DEFAULT_MIN_ABSTENTION, help="Minimum abstention rate on absent facts with RAG on (default 0.75).")
    parser.add_argument("--reference", action="store_true", help="Also write the result to benchmarks/reference/rag_eval_<model>.json.")
    parser.add_argument("--quiet", action="store_true", help="Do not print one progress line per answer.")
    parser.add_argument("--rescore", type=Path, metavar="RESULT_JSON",
                        help="Do not call the model: score the answers saved in RESULT_JSON again with the current eval set.")
    arguments = parser.parse_args(argv)
    if arguments.runs is None:
        arguments.runs = 1 if arguments.quick else DEFAULT_RUNS
    if arguments.runs < 1:
        parser.error("--runs must be at least 1")
    if arguments.max_answer_tokens < 0:
        parser.error("--max-answer-tokens must be 0 or more")
    if arguments.temperature is not None and not 0 <= arguments.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    if arguments.rescore:
        try:
            return rescore(arguments.rescore, arguments.eval_set, arguments.min_gain, arguments.min_abstention)
        except (common.SetupError, OSError, ValueError, KeyError) as error:
            print(f"Setup error: {error}")
            return common.EXIT_SETUP_ERROR
    started = perf_counter()
    try:
        eval_set = load_eval_set(arguments.eval_set)
        questions = select_questions(eval_set, arguments.quick)
        documents = corpus_paths(eval_set)
        harness = common.AppHarness(model=arguments.model, temperature=arguments.temperature,
                                    with_teacher_mcp=arguments.with_mcp, label="rag-eval",
                                    max_answer_tokens=arguments.max_answer_tokens)
        common.check_models(harness.config)
    except (common.SetupError, OSError, ValueError) as error:
        print(f"Setup error: {error}")
        return common.EXIT_SETUP_ERROR

    settings = harness.describe()
    counts = {kind: sum(q["type"] == kind for q in questions) for kind in ("answerable", "absent", "control")}
    print(f"SensAI RAG evaluation ({eval_set['version']}{', quick subset' if arguments.quick else ''})")
    print(f"Model {settings['chat_model']} | temperature {settings['temperature']} | num_ctx {settings['context_window']} | "
          f"embeddings {settings['embedding_model']} | top_k {settings['rag_top_k']} | threshold {settings['rag_relevance_threshold']}")
    print(f"Questions: {counts['answerable']} answerable, {counts['absent']} absent, {counts['control']} control | "
          f"{arguments.runs} run(s) x 2 conditions = {len(questions) * 2 * arguments.runs} answers")
    print(f"Pipeline: {settings['pipeline']}; isolated temporary workspace; "
          f"MCP servers {'teacher only' if arguments.with_mcp else 'disabled'}; network tools disabled.")
    if arguments.max_answer_tokens:
        print(f"Replies longer than {arguments.max_answer_tokens} tokens are cut and scored 'runaway' (the app itself sets no limit).")
    ollama = common.ollama_metadata(harness.config.model.base_url)
    if ollama.get("loaded_models_at_start"):
        print(f"Note: models already loaded in Ollama at start: {', '.join(ollama['loaded_models_at_start'])} (shared server).")

    try:
        with harness:
            if arguments.persona:
                output = harness.command(f"/persona use {arguments.persona}")
                if "changed to" not in output:
                    raise common.SetupError(output)
            settings["persona"] = harness.app.session.active_persona_id
            harness.ingest(documents)
            indexed = harness.indexed_documents()
            print(f"Corpus indexed with /ingest: {len(indexed)} documents, {sum(count for _, count in indexed)} chunks.\n")
            results = run_evaluation(harness, questions, arguments.runs, progress=(lambda _line: None) if arguments.quiet else print)
    except common.SetupError as error:
        print(f"Setup error: {error}")
        return common.EXIT_SETUP_ERROR

    summary = summarize(results, arguments.runs)
    result_verdict = verdict(summary, arguments.min_gain, arguments.min_abstention)
    elapsed = perf_counter() - started
    print(format_report(summary, results, questions, result_verdict, arguments.runs))

    output = {
        "kind": "sensai-rag-evaluation",
        "eval_set": eval_set["version"],
        "created_at": common.utc_timestamp(),
        "git_commit": common.git_commit(),
        "git_tracked_changes": common.git_dirty(),
        "duration_seconds": round(elapsed, 1),
        "quick": arguments.quick,
        "runs": arguments.runs,
        "settings": settings,
        "ollama": ollama,
        "machine": common.machine_metadata(),
        "corpus": [{"path": common.relative(path), "sha256_12": common.file_digest(path)} for path in documents],
        "eval_set_sha256_12": common.file_digest(arguments.eval_set),
        "thresholds": {"min_gain_points": arguments.min_gain, "min_abstention": arguments.min_abstention},
        "summary": summary,
        "verdict": result_verdict,
        "results": results,
    }
    slug = common.model_slug(settings["chat_model"])
    stamp = output["created_at"].replace(":", "").replace("-", "")
    saved = common.write_json(common.RESULTS_DIR / f"rag_eval_{stamp}_{slug}{'_quick' if arguments.quick else ''}.json", output)
    print(f"\nFull results (every answer): {common.relative(saved)}  |  duration {elapsed / 60:.1f} min")
    if arguments.reference:
        reference = common.write_json(common.REFERENCE_DIR / f"rag_eval_{slug}.json", output)
        print(f"Reference copy: {common.relative(reference)}")
    return common.EXIT_PASS if result_verdict["passed"] else common.EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
