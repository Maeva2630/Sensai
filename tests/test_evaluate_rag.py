"""Offline tests for the RAG A/B evaluation script and its shared harness (no Ollama needed)."""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_common as common  # noqa: E402
import evaluate_rag  # noqa: E402
from mcp.servers.teacher_tools import calculate_grade, class_statistics  # noqa: E402

REAL_HARNESS = common.AppHarness
EVAL_SET = json.loads((ROOT / "benchmarks" / "eval" / "rag_eval_fr.json").read_text(encoding="utf-8"))
QUESTIONS = {question["id"]: question for question in EVAL_SET["questions"]}


class TestNormalisation(unittest.TestCase):
    def test_accents_case_and_thin_spaces_are_unified(self):
        self.assertEqual(common.normalize("Le  Contrôle a lieu le 17 Décembre"), "le controle a lieu le 17 decembre")

    def test_hours_fractions_and_decimals_share_one_notation(self):
        self.assertEqual(common.normalize("à 10 h 30"), "a 10h30")
        self.assertEqual(common.normalize("à 12:40"), "a 12h40")
        self.assertEqual(common.normalize(r"soit \frac{3}{4} ou \dfrac{23}{20}"), "soit 3/4 ou 23/20")
        self.assertEqual(common.normalize("3 / 4"), "3/4")
        self.assertEqual(common.normalize("perd 0.5 point"), "perd 0,5 point")

    def test_a_spaced_colon_is_not_read_as_an_hour(self):
        self.assertEqual(common.normalize("Exercice 1 : 12 et 14"), "exercice 1 : 12 et 14")

    def test_markdown_emphasis_and_typographic_signs_are_removed(self):
        self.assertEqual(common.normalize("**L’erreur « D+D »** − fin"), "l'erreur \" d+d \" - fin")


class TestKeywordScoring(unittest.TestCase):
    def test_literal_variants_match_whole_words_only(self):
        self.assertTrue(common.matches_variant("en salle b12.", "B12"))
        self.assertFalse(common.matches_variant("en salle b12.", "12"))
        self.assertFalse(common.matches_variant("110,5 points", "0,5"))

    def test_coefficient_answer_from_the_baseline_run_is_accepted(self):
        groups = QUESTIONS["A04"]["expected"]
        self.assertTrue(common.is_correct("Le coefficient du contrôle commun sur les fractions dans la moyenne du trimestre est de 2.", groups))
        self.assertFalse(common.is_correct("1. question 1 : calcul. 2. question 2 : coefficient du controle commun.", groups))

    def test_regex_variants_apply_to_the_normalised_answer(self):
        self.assertTrue(common.matches_variant(common.normalize("Il dure 20 minutes."), r"re:\b20 ?(min|minutes)\b"))

    def test_every_group_must_match(self):
        groups = QUESTIONS["A06"]["expected"]
        self.assertTrue(common.is_correct("Les exercices n°12 et n°14 page 87.", groups))
        self.assertFalse(common.is_correct("Les exercices n°12 et n°14.", groups))
        self.assertFalse(common.is_correct("anything", []))

    def test_reference_answers_of_the_eval_set_score_as_correct(self):
        for question in EVAL_SET["questions"]:
            if question["type"] == "absent":
                continue
            with self.subTest(question=question["id"]):
                self.assertTrue(common.is_correct(question["answer"], question["expected"]))

    def test_answerable_questions_point_to_a_corpus_document(self):
        corpus = {Path(entry).name for entry in EVAL_SET["corpus"]}
        for entry in EVAL_SET["corpus"]:
            self.assertTrue((ROOT / entry).is_file(), entry)
        for question in EVAL_SET["questions"]:
            if question["type"] == "answerable":
                self.assertIn(question["source"], corpus, question["id"])

    def test_the_eval_set_measures_the_time_tool_routing(self):
        tagged = [q for q in EVAL_SET["questions"] if "time-tool-keyword" in q.get("tags", [])]
        self.assertTrue(any("date" in q["question"] for q in tagged))


class TestAbstentionAndOutcomes(unittest.TestCase):
    def test_abstention_markers_in_both_languages(self):
        self.assertTrue(common.is_abstention("I cannot find this information in the indexed documents."))
        self.assertTrue(common.is_abstention("Je ne trouve pas cette information dans les documents."))
        self.assertTrue(common.is_abstention("Le document n'est pas précis : la date n’est pas mentionnée."))
        self.assertFalse(common.is_abstention("Le contrôle a lieu en salle B12."))

    def test_answerable_outcomes(self):
        question = QUESTIONS["A01"]
        self.assertEqual(evaluate_rag.score_answer(question, "En salle B12.")["outcome"], "correct")
        self.assertEqual(evaluate_rag.score_answer(question, "Je ne trouve pas cette information.")["outcome"], "abstained")
        self.assertEqual(evaluate_rag.score_answer(question, "En salle B07.")["outcome"], "wrong")

    def test_a_runaway_reply_is_never_correct(self):
        score = evaluate_rag.score_answer(QUESTIONS["A01"], "salle B12 " * 50, runaway=True)
        self.assertEqual(score["outcome"], "runaway")
        self.assertFalse(score["correct"])

    def test_absent_outcomes(self):
        question = QUESTIONS["N03"]
        abstained = evaluate_rag.score_answer(question, "I cannot find this information in the indexed documents.")
        invented = evaluate_rag.score_answer(question, "La sortie coûte 12 euros par élève.")
        evasive = evaluate_rag.score_answer(question, "Voici une activité sur les fractions pour vos élèves.")
        hedged = evaluate_rag.score_answer(question, "Je ne trouve pas le prix, mais comptez 8 €.")
        self.assertEqual((abstained["outcome"], abstained["abstained"]), ("abstained", True))
        self.assertEqual((invented["outcome"], invented["hallucinated"]), ("hallucinated", True))
        self.assertEqual(evasive["outcome"], "evasive")
        self.assertTrue(hedged["abstained"] and hedged["hedged_guess"])


def _row(condition, kind, run=1, correct=False, abstained=False, outcome=None, source=None):
    return {
        "run": run, "condition": condition, "type": kind, "id": "X", "source": source,
        "correct": correct if kind != "absent" else None, "abstained": abstained,
        "hallucinated": kind == "absent" and not abstained,
        "outcome": outcome or ("correct" if correct else "abstained" if abstained else "wrong"),
        "rag_context_used": condition == "on", "retrieved_sources": [source] if source else [],
        "tool_calls": [], "tool_names": [], "latency_ms": 1000.0, "first_token_ms": 100.0,
        "prompt_tokens": 100, "completion_tokens": 20,
    }


class TestSummaryAndVerdict(unittest.TestCase):
    def _summary(self, off_correct, on_correct, on_abstained, total=10, absent=4):
        rows = []
        for index in range(total):
            rows.append(_row("off", "answerable", correct=index < off_correct))
            rows.append(_row("on", "answerable", correct=index < on_correct, source="doc.md"))
        for index in range(absent):
            rows.append(_row("off", "absent", abstained=True))
            rows.append(_row("on", "absent", abstained=index < on_abstained))
        return evaluate_rag.summarize(rows, runs=1)

    def test_summary_counts_each_condition(self):
        summary = self._summary(off_correct=1, on_correct=8, on_abstained=3)
        self.assertAlmostEqual(summary["off"]["accuracy"], 0.1)
        self.assertAlmostEqual(summary["on"]["accuracy"], 0.8)
        self.assertEqual(summary["on"]["absent_hallucinated"], 1)
        self.assertEqual(summary["on"]["retrieval_hits"], 10)

    def test_pass_when_gain_and_abstention_reach_their_thresholds(self):
        result = evaluate_rag.verdict(self._summary(off_correct=1, on_correct=4, on_abstained=3), 30, 0.75)
        self.assertTrue(result["passed"])
        self.assertEqual(result["line"], "VERDICT: PASS")
        self.assertIn("off 10% → on 40% (+30 pts, threshold +30)  PASS", result["checks"][0]["line"])

    def test_fail_when_the_gain_is_too_small(self):
        result = evaluate_rag.verdict(self._summary(off_correct=2, on_correct=4, on_abstained=4), 30, 0.75)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"][0]["line"].endswith("FAIL"))

    def test_fail_when_rag_on_does_not_abstain_on_absent_facts(self):
        result = evaluate_rag.verdict(self._summary(off_correct=0, on_correct=9, on_abstained=2), 30, 0.75)
        self.assertFalse(result["passed"])
        self.assertIn("2/4 (50%, threshold 75%)  FAIL", result["checks"][1]["line"])


class FakeHarness:
    """Answers like a model that knows the corpus only when RAG is on."""

    def __init__(self, *, rag_helps=True, **_options):
        self.rag_helps = rag_helps
        self.rag = False
        self.config = REAL_HARNESS().config
        self.app = type("App", (), {"session": type("Session", (), {"active_persona_id": "teacher-assistant"})()})()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def describe(self):
        return {"pipeline": "fake", "chat_model": "fake:model", "temperature": 0.2, "context_window": 4096,
                "embedding_model": "fake-embed", "rag_top_k": 3, "rag_relevance_threshold": 0.35}

    def command(self, _text):
        return ""

    def ingest(self, paths):
        return f"Multiple ingestion complete: {len(paths)}/{len(paths)} paths indexed."

    def indexed_documents(self):
        return [("doc.md", 1)]

    def set_rag(self, enabled):
        self.rag = enabled

    def ask(self, question):
        item = next(q for q in EVAL_SET["questions"] if q["question"] == question)
        if item["type"] == "absent" or not (self.rag and self.rag_helps):
            answer = "Je ne trouve pas cette information."
        else:
            answer = item["answer"]
        return common.TurnRecord(question=question, answer=answer, latency_ms=10.0, rag_context_used=self.rag,
                                 retrieved=[{"source": item.get("source") or "doc.md", "score": 0.8}] if self.rag else [])


class TestMainExitCodes(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        for name, target in (("RESULTS_DIR", self.folder / "results"), ("REFERENCE_DIR", self.folder / "reference")):
            patcher = patch.object(common, name, target)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(common, "ollama_metadata", return_value={"version": None, "loaded_models_at_start": []})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, harness_options, *arguments):
        output = io.StringIO()
        with patch.object(common, "AppHarness", lambda **options: FakeHarness(**{**options, **harness_options})), \
                patch.object(common, "check_models"), redirect_stdout(output):
            code = evaluate_rag.main(["--quiet", *arguments])
        return code, output.getvalue()

    def test_exit_code_zero_and_verdict_when_rag_helps(self):
        code, output = self._run({"rag_helps": True}, "--quick", "--reference")
        self.assertEqual(code, common.EXIT_PASS)
        self.assertIn("VERDICT: PASS", output)
        saved = json.loads(next((self.folder / "results").glob("rag_eval_*.json")).read_text(encoding="utf-8"))
        self.assertTrue(saved["verdict"]["passed"])
        self.assertTrue(all("answer" in row for row in saved["results"]))
        self.assertTrue((self.folder / "reference" / "rag_eval_fake-model.json").is_file())

    def test_exit_code_one_when_rag_brings_no_gain(self):
        code, output = self._run({"rag_helps": False}, "--runs", "1")
        self.assertEqual(code, common.EXIT_FAIL)
        self.assertIn("VERDICT: FAIL", output)

    def test_rescore_applies_the_current_eval_set_to_saved_answers(self):
        self._run({"rag_helps": True}, "--quick")
        saved = next((self.folder / "results").glob("rag_eval_*.json"))
        data = json.loads(saved.read_text(encoding="utf-8"))
        for row in data["results"]:
            if row["condition"] == "on" and row["type"] == "answerable":
                row["answer"] = "Je ne trouve pas cette information."
        saved.write_text(json.dumps(data), encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            code = evaluate_rag.main(["--rescore", str(saved)])
        self.assertEqual(code, common.EXIT_FAIL)
        rescored = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(rescored["rescoring"][0]["verdict_before"], "VERDICT: PASS")
        self.assertEqual(rescored["verdict"]["line"], "VERDICT: FAIL")

    def test_exit_code_two_when_ollama_is_unavailable(self):
        output = io.StringIO()
        with patch.object(common, "check_models", side_effect=common.SetupError("Ollama is not reachable")), \
                redirect_stdout(output):
            code = evaluate_rag.main(["--quick"])
        self.assertEqual(code, common.EXIT_SETUP_ERROR)
        self.assertIn("Setup error: Ollama is not reachable", output.getvalue())

    def test_quick_subset_has_no_control_question(self):
        questions = evaluate_rag.select_questions(EVAL_SET, quick=True)
        self.assertTrue(questions)
        self.assertTrue(all(q.get("quick") and q["type"] != "control" for q in questions))
        self.assertTrue(any(q["type"] == "absent" for q in questions))


class TestHarnessObservation(unittest.TestCase):
    """The harness records Ollama's counters and cuts runaway replies without altering lines."""

    def _client(self, lines):
        class Client:
            def stream_message(self, messages, tools=None, options=None):
                yield from lines

            def complete_message(self, messages, tools=None, output_format=None):
                return type("Response", (), {"content": json.dumps({"done": True, "prompt_eval_count": 7, "eval_count": 3})})()
        return Client()

    def test_final_counters_are_recorded_and_lines_pass_through(self):
        lines = [json.dumps({"message": {"content": "Bonjour"}, "done": False}).encode(),
                 json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 2,
                             "total_duration": 5_000_000, "load_duration": 2_000_000_000}).encode()]
        harness = REAL_HARNESS()
        client = self._client(lines)
        harness._instrument(client)
        self.assertEqual(list(client.stream_message([{"role": "user", "content": "Salut"}])), lines)
        call = harness.model_calls[0]
        self.assertEqual((call["prompt_eval_count"], call["eval_count"], call["total_duration_ms"]), (12, 2, 5.0))
        record = common.TurnRecord(question="q", model_calls=[call])
        self.assertTrue(record.model_reloaded)
        client.complete_message([])
        self.assertEqual(harness.model_calls[1]["prompt_eval_count"], 7)

    def test_a_reply_longer_than_the_cap_is_cut_and_flagged(self):
        lines = [json.dumps({"message": {"content": "x"}, "done": False}).encode() for _ in range(50)]
        harness = common.AppHarness(max_answer_tokens=10)
        client = self._client(lines)
        harness._instrument(client)
        received = list(client.stream_message([]))
        self.assertEqual(len(received), 11)
        self.assertTrue(common.TurnRecord(question="q", model_calls=harness.model_calls).runaway)

    def test_retrieved_sources_are_read_from_the_rag_message(self):
        context = ("[Source: progression_6e_B.md | Chunk: abc | Lines: 1-9 | Score: 0.753]\ntext\n\n"
                   "[Source: consignes_evaluation.md | Chunk: def | Lines: 2-3 | Language: markdown | Score: 0.700]\nmore")
        self.assertEqual([item["source"] for item in common.parse_retrieved(context)],
                         ["progression_6e_B.md", "consignes_evaluation.md"])
        self.assertEqual(common.parse_retrieved(None, "Sources: a.md (chunks: 1); b.md (chunks: 2)")[1]["source"], "b.md")

    def test_only_warnings_shown_before_the_answer_count_as_a_guard(self):
        record = common.TurnRecord(question="q", notices=["Warning: Context token usage is at 85.0%"])
        self.assertEqual(common.AppHarness.guard_messages(record), [])
        record.pre_send_messages = ["Message trop long pour la fenêtre de contexte (4096 tokens)."]
        self.assertEqual(len(common.AppHarness.guard_messages(record)), 1)

    def test_the_harness_runs_in_a_temporary_directory_and_restores_the_working_directory(self):
        before = os.getcwd()
        with patch.object(common, "WebApp") as web_app, patch.object(common, "configure_logging"):
            web_app.return_value.runtime.rag_indexer.vector_store.database_path = "data/sensai_rag.sqlite3"
            harness = common.AppHarness()
            with harness:
                self.assertNotEqual(os.getcwd(), before)
                self.assertTrue((harness.workspace / "profile.json").is_file())
                workspace = harness.workspace
        self.assertEqual(os.getcwd(), before)
        self.assertFalse(workspace.exists())


class TestGradesDataset(unittest.TestCase):
    """The fictional grade file is directly usable with the teacher MCP tools."""

    def setUp(self):
        text = (ROOT / "courses" / "notes_6e_B.csv").read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO("\n".join(line for line in text.splitlines() if not line.startswith("#")))))
        self.header, self.coefficients, self.pupils = rows[0], rows[1], rows[2:]

    def test_marks_and_coefficients_feed_calculate_grade(self):
        coefficients = [float(value) for value in self.coefficients[1:]]
        self.assertEqual(coefficients, [1.0, 1.0, 2.0])
        ambre = next(row for row in self.pupils if row[0] == "Ambre")
        result = calculate_grade({"marks": [float(value) for value in ambre[1:]], "coefficients": coefficients})
        self.assertEqual(result, "Weighted average: 13.5/20 (computed from 3 marks)")

    def test_a_column_feeds_class_statistics(self):
        marks = [float(row[3]) for row in self.pupils]
        self.assertEqual(len(marks), 26)
        self.assertIn("Students: 26", class_statistics({"marks": marks}))


if __name__ == "__main__":
    unittest.main()
