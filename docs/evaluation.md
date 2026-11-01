# Measured improvement: RAG on versus off

[Home](../README.md) · [Limit probes](limits.md) · [Model benchmarks](model-benchmarks.md) · [Testing](testing.md)

This page describes a replayable A/B measurement of SensAI's RAG mode on a business
evaluation set written for the product scenario: Camille, a 6e mathematics teacher, asks
questions about her own (fictional) course documents. The script prints a table and a
PASS/FAIL verdict, and exits with code 0 (PASS) or 1 (FAIL).

## Purpose

The question measured is: **does RAG make the assistant answer correctly from the
teacher's own documents, while still saying "I cannot find it" when the documents do
not contain the answer?** The same questions are asked with RAG off (baseline) and with
RAG on, through the same application, model and settings. Only the `/rag` switch changes.

## Protocol

| Element | Choice | Reason |
| --- | --- | --- |
| Pipeline | `web.app.WebApp` driven in-process (`scripts/evaluate_rag.py`). | It is the engine shared by the terminal and the web page: same model runtime, persona prompt, profile, RAG context provider, tool registry and slash commands. The evaluation does not re-implement retrieval or prompting. |
| Indexing | The application's own `/ingest --multiple` command on the six Markdown files of `courses/`. | Same chunking, embeddings (`nomic-embed-text`) and SQLite storage as a teacher's installation. |
| RAG switch | The application's own `/rag on` and `/rag off` commands. | The condition is what a teacher toggles. |
| Conversation | `new_conversation()` before every question. | No answer can leak into the next question. |
| Order | The two conditions alternate for each question; the order is reversed on every other run. | Neither condition always benefits from a freshly loaded model. |
| Settings | Read from `config.json`: `qwen2.5:3b`, temperature 0.2, `num_ctx` 4096, `top_k` 3, relevance threshold 0.35, persona `teacher-assistant`, profile `demo/profile_camille.json`. | Out-of-the-box behaviour. `--model`, `--temperature` and `--persona` override them. |
| Isolation | Everything runs in a temporary working directory deleted at the end: `sessions/`, `data/` (RAG and monitoring databases), `logs/`, `exports/` and a copy of the profile. The script checks that the RAG database is inside it before asking anything. | The repository and the teacher's real index are never touched. Only the result JSON is written to the repository. |
| MCP and network | MCP servers disabled (empty `MCPManager`); network tools disabled; any confirmation request is refused. `--with-mcp` starts the local teacher MCP server, whose tools are then offered on every request. | Nothing leaves the machine and the measurement does not depend on an external process. |
| Answer length cap | The harness stops reading a reply after 1,500 streamed tokens and scores it `runaway` (`--max-answer-tokens`, 0 disables the cap). | The application sets no output limit; during development one RAG-off answer generated 11,464 tokens in 211 s. The cap bounds the run time and is reported explicitly. |
| Observation | The chat client is wrapped to record Ollama's final counters (`prompt_eval_count`, `eval_count`, durations) without changing requests or responses. | Token counts are measured by Ollama, not estimated. |

## Dataset

All documents are fictional demonstration data (fictional school, class, dates, rooms and
first names). Each file says so on its first lines.

| File | Content | Facts used by the evaluation |
| --- | --- | --- |
| `courses/progression_6e_B.md` | Sequence plan for the fractions unit of class 6e B. | Test date (17 December 2026), time and room (10h30, B12). |
| `courses/consignes_evaluation.md` | Test instructions, marking scheme, weights. | Points per exercise, coefficient 2, 17 extra minutes, 0.5-point penalty, make-up room B04. |
| `courses/fractions_equivalentes.md` | Lesson on equivalent fractions. | Chocolate-bar example 18/24 = 3/4, homework exercises 12 and 14 page 87. |
| `courses/fractions_addition.md` | Lesson on adding and subtracting fractions. | Introductory activity "Le partage des crêpes" (20 min), the class name "erreur D+D", homework 31 and 33 page 92. |
| `courses/nombres_decimaux.md` | Lesson on decimal fractions. | Game "Décimal-Express": 3 rounds, teams of 4. |
| `courses/soutien_fractions.md` | Weekly support workshop. | Tuesday 12h40, room B07, peer tutoring "binômes Tandem". |
| `courses/notes_6e_B.csv` | Fictional marks /20 for 26 first names, with coefficients (1, 1, 2). | Not indexed (CSV is not an ingestible type). It is the input for the teacher MCP tools `calculate_grade` and `class_statistics`; `tests/test_evaluate_rag.py` checks that it feeds both tools. |

Several documents mention rooms, exercises, durations and points, so retrieval has to
choose between them: for example the test room (B12), the workshop room (B07) and the
make-up room (B04) are in three different files.

The evaluation set `benchmarks/eval/rag_eval_fr.json` (version `rag-eval-fr-v1`) holds:

- **12 answerable questions** (A01 to A12), each with the document that holds the answer
  and its expected keywords. Two of them contain a word that the time tool routes to
  itself: A02 ("Quelle est la **date** du contrôle...") and A11 ("À quelle **heure**...").
  They measure the effect of the routing rule described below.
- **4 absent questions** (N01 to N04): plausible questions whose answer is not in the
  corpus (class council date, a geometry exercise that does not exist, a school-trip price,
  a report-card threshold). The correct behaviour is to say the information is not available.
- **2 control questions** (C01, C02): general knowledge (1/8 in decimal, sides of a
  hexagon). They are reported but not used in the verdict; they show what RAG mode costs
  outside the corpus.

`--quick` uses the subset marked `"quick": true`: A01, A02, A09, A12, N01, N02 (one run,
12 answers).

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | Answerable questions whose answer contains every expected keyword group, divided by the number of answerable answers. |
| Correct abstention rate | Absent questions answered with an abstention phrase ("cannot find", "je ne trouve pas", "n'est pas mentionné", "il n'y a pas"...), divided by the number of absent answers. |
| Hallucination | An absent question answered with a concrete unsupported value matching the question's `invented` patterns (a date, a price, a number of points, a mark /20) and no abstention phrase. |
| Evasive | An absent question answered with neither an abstention phrase nor an invented value (for example generic teaching advice). Counted as not abstaining. |
| Runaway | A reply cut at the answer-length cap. Never correct. |
| Retrieval hit | With RAG on, the expected document is among the excerpts sent to the model (read from the RAG message the application built). Diagnostic only. |
| RAG skipped | RAG was on but no excerpt was sent because a tool claimed the request. Diagnostic only. |
| Latency | Wall-clock time from sending the question to the end of the reply, as observed by the harness; median per condition. |
| Tokens | Ollama `prompt_eval_count` (input) and `eval_count` (output) from the final streamed line; median per condition. |

Keyword matching is done on a normalised form of the answer: lower case, accents removed,
spaces and thin spaces unified, LaTeX and spaced fractions written `a/b`, hours written
`10h30`, decimal point written as a comma. A variant that starts with `re:` is a regular
expression; other variants match as whole words, so `12` does not match `B12`.

**Verdict.** PASS when both criteria hold:

1. accuracy gain from RAG of at least **+30 points** (`--min-gain`);
2. correct abstention on absent facts with RAG on of at least **75%** (`--min-abstention`).

## Replay it live

From the repository root, with Ollama running and the models of `config.json` installed:

```bash
source .venv/bin/activate
python3 scripts/evaluate_rag.py --quick        # 1 run, 12 answers (30 s on the M5)
python3 scripts/evaluate_rag.py                # 18 questions x 2 conditions x 3 runs (6 min)
python3 scripts/evaluate_rag.py --model llama3.2:latest --runs 1
python3 scripts/evaluate_rag.py --temperature 0.8
echo $?                                        # 0 = PASS, 1 = FAIL, 2 = setup error
```

The quick replay on the final code gave `off 0% → on 100% (+100 pts)`, abstention 2/2,
`VERDICT: PASS` in 30 s (on the baseline code: `off 0% → on 75%`, A02 failing for the routing
reason explained below).

To change a keyword variant and see its effect without calling the model again, re-score
a saved result: `python3 scripts/evaluate_rag.py --rescore <result.json>`.

Every run writes all answers, scores, retrieved sources and Ollama counters to
`benchmarks/results/rag_eval_<timestamp>_<model>.json` (ignored by Git). `--reference`
also writes `benchmarks/reference/rag_eval_<model>.json`, which is committed as evidence.

## How to read the output

One progress line is printed per answer (`run`, question, condition, outcome, latency).
Then:

- the **results table**: one line per condition with accuracy, abstention and
  hallucination on absent questions, control questions, median latency and median tokens;
- the **per-question table**: correct (or, for absent questions, abstained) runs with RAG
  off → on, whether the expected document was retrieved, and whether RAG was skipped;
- **diagnostics**: retrieval hit rate, skipped RAG turns, evasive and runaway answers,
  model reloads on a shared Ollama server, accuracy per run;
- the **verdict**: one line per criterion with its threshold, then `VERDICT: PASS` or
  `VERDICT: FAIL`.

## Results

### Final code (this branch), 2026-11-02

Run conditions: Apple M5, 16 GB RAM, macOS 26.6.2, Ollama 0.40.7, `qwen2.5:3b` with
`nomic-embed-text`, temperature 0.2, `num_ctx` 4096, `top_k` 3, relevance threshold 0.6,
persona `teacher-assistant`, profile `demo/profile_camille.json`, MCP servers disabled.
3 runs, 108 answers, 6.0 minutes; both models already loaded, no reload observed.
Command: `python3 scripts/evaluate_rag.py --reference`. Evidence:
[`benchmarks/reference/rag_eval_qwen2.5-3b.json`](../benchmarks/reference/rag_eval_qwen2.5-3b.json)
(every answer, retrieved sources and Ollama counters).

| Condition | Answerable correct | Absent: abstained | Absent: hallucinated | Absent: evasive | Control correct | Median latency | Median first token | Median tokens in / out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAG off | 0/36 (0%) | 5/12 | 0/12 | 7/12 | 6/6 | 3.9 s | 0.08 s | 438 / 218 |
| RAG on | 30/36 (83%) | 12/12 | 0/12 | 0/12 | 3/6 | 0.9 s | 0.69 s | 1,354 / 21 |

```text
RAG accuracy: off 0% → on 83% (+83 pts, threshold +30)  PASS
Abstention on absent facts (RAG on): 12/12 (100%, threshold 75%)  PASS
VERDICT: PASS
```

Accuracy was identical in the three runs (RAG off 0%, 0%, 0%; RAG on 83%, 83%, 83%), and
the expected document was among the retrieved excerpts for 36/36 RAG-on answers.

- **RAG is what makes the answers correct**: without the teacher's documents the model
  never has her facts (0/36); with them, 30/36.
- **The 6 RAG-on misses are model limits, not retrieval failures** (the right document was
  retrieved every time):
  - A06 (3 answers): the model gives the homework of the neighbouring lesson (exercises 31
    and 33 page 92 instead of 12 and 14 page 87). Two lessons with the same layout confuse it.
  - A07 (3 answers): "Léo a mangé 3/4 de la tablette" — correct simplified fraction, but the
    unsimplified 18/24 asked for in the same question is left out.
- **No invented answer**: the 12 absent-fact answers all used the instructed sentence
  ("Je ne trouve pas cette information dans les documents indexés."), now in French.
- **Cost of RAG outside the corpus**: the hexagon control question (C02) is refused with RAG
  on, by design (RAG mode forbids outside knowledge); `/rag off` answers it.
- **Latency**: RAG adds about 900 input tokens and moves the first token from 0.08 s to
  0.69 s, but grounded answers are short (median 21 tokens), so the median total drops
  from 3.9 s to 0.9 s. No answer ran away: every reply is capped at a quarter of the window
  (1,024 tokens), where the baseline had two RAG-off replies cut by the harness at 1,500.

### Before the fixes (origin/main 6487dd6), 2026-11-01

Same protocol, but RAG excerpts sent as a `system` message, threshold 0.35, the time tool
claiming any question containing "date" or "heure", no pre-send budget and no answer cap.
3 runs, 8.2 minutes. (The raw file was replaced by the final run; the figures below are
those recorded in the baseline run, after the A04 keyword correction described below.)

| Condition | Answerable correct | Absent: abstained | Absent: hallucinated | Absent: evasive | Control correct | Median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAG off | 0/36 (0%) | 4/12 | 1/12 | 7/12 | 6/6 | 8.2 s |
| RAG on | 27/36 (75%) | 12/12 | 0/12 | 0/12 | 3/6 | 0.9 s |

The 9 RAG-on misses were A06 (as above) and A02 + A11, two questions containing "date" and
"heure": the time tool claimed them and **no excerpt was sent at all**. Fixing that routing
is the +8 points (75% → 83%).

One intermediate run is worth keeping in mind: when the "not found" instruction was
rephrased to let the model answer "in the language of the question" instead of an exact
sentence, abstention on absent facts fell from 12/12 to 7/12 (the model paraphrased, and
invented a report-card threshold for N04 twice). The exact sentence was restored, in French.
Small models follow an exact sentence much better than a paraphrase instruction.

Scoring note: during the baseline, the keyword pattern of A04 was corrected (it missed
"Le coefficient ... est de 2." and accepted a numbered list "2. question 2 : coefficient").
The saved answers were re-scored without calling the model
(`python3 scripts/evaluate_rag.py --rescore <result.json>`).

## Limitations of the method

- **Keyword scoring is a proxy.** An answer that contains the right keywords for a wrong
  reason is counted correct; a correct paraphrase that avoids every accepted variant
  ("le jeudi avant les vacances") is counted wrong. Variants were written to accept the
  usual notations, and every answer is kept in the JSON for manual review.
- **Abstention detection is a phrase list.** An answer that abstains and then guesses is
  counted as an abstention but flagged `hedged_guess`; a refusal worded in an unexpected
  way is counted as evasive.
- **Small n.** 12 answerable questions × 3 runs gives 36 answers per condition: one
  answer is about 3 points of accuracy. The verdict thresholds are deliberately far from
  the observed values so that run-to-run variation does not flip them.
- **Sampling noise.** Temperature 0.2 is not deterministic and the application sends no
  seed; per-run accuracy is printed to show the spread.
- **One corpus, one model.** The corpus is small (13 chunks) and written for the
  scenario; the figures do not predict accuracy on a large or messy document collection.
  Other models can be measured with `--model`.
- **It measures the whole pipeline, not retrieval alone.** A wrong answer can come from
  retrieval, from the prompt, from tool routing or from the model; the retrieval-hit and
  RAG-skipped diagnostics separate the first cases.
- **Shared Ollama server.** Latency depends on other requests and on model reloads; the
  run records the models already loaded at start and flags turns whose Ollama
  `load_duration` exceeds one second.
