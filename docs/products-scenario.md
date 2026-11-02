# Product scenario

[Home](../README.md) · [User stories](user-stories.md) · [Technical decisions](decisions.md) · [Demo script](../demo/demo-script.md) · [Model benchmarks](model-benchmarks.md)

## Product vision

SensAI is a local assistant for teachers preparing learning activities, classroom materials, and assessments. It consults local documents, adapts activities, generates quizzes, and checks routine calculations while the teacher retains control of the final material and local data. Generated answers, sources, grades, and exports must always be reviewed before use with learners.

## Target users

The recommended deployment scope is **primary and middle-school teachers** (tickets #124 and #131).

| Audience | Typical uses | Product positioning |
| --- | --- | --- |
| Primary-school teachers | Clear activities, short quizzes, and age-appropriate explanations from local resources. | The teacher checks suitability for pupils and validates generated content. |
| Middle-school teachers | Subject resources, assessments, grade calculations, and longer preparation sessions. | SensAI only knows a curriculum supplied in local documents. |
| High-school teachers | The same features are technically available. | The model choice, prompts, and scenario were designed and checked for primary and middle school only; high-school use needs its own evaluation. |

All implemented features are available to every audience; only the validation scope differs.

## Persona: Camille

The scenario follows one teacher from start to finish. The live demonstration uses the same person and the same sequence: [`demo/profile_camille.json`](../demo/profile_camille.json) and the fictional 6e B course in [`courses/`](../courses/).

| | Camille | Status |
| --- | --- | --- |
| Role | Mathematics teacher in a French public *collège*, teaching *sixième* (first year of *collège*, pupils aged 11–12). | Scenario choice, matches the demo profile |
| Classes | Four *sixième* classes, about 100 pupils. | Assumption |
| Language | Works and writes in French. | Demo profile (`preferred_language: French`) |
| Constraints she gives the assistant | "Use examples appropriate for sixth-grade pupils", "Prefer concise classroom-ready activities". | Demo profile (`constraints`) |
| Where she prepares | At home in the evening, and in the staff room between lessons, where the school network is filtered and unreliable. | Assumption |
| Computer | Her own laptop: Apple M5, 16 GB unified memory. | The team's measurement machine; see [Hardware feasibility](#hardware-feasibility) |
| Material she owns | Course notes, worksheets, and past tests in Markdown, PDF, and Word files. | Assumption; these formats are supported by `/ingest` |
| Pupil data she handles | Marks, difficulties, and first names of minors. | Fact of the job; see [local data and GDPR assessment](RGPD.md) |

### Business need

The figures below are **assumptions** used to size the scenario. They have not been measured with real teachers.

| Item | Value |
| --- | --- |
| Preparation time outside class | About 6 hours per week. |
| What she prepares per chapter (about two weeks) | One lesson plan, about four 15-minute starter activities, one 5-question multiple-choice check with a pupil copy and a teacher copy. |
| What she computes per term | Weighted averages and class statistics for about 100 pupils. |
| Target gain | 1 hour per week (about 15 % of preparation time): she reviews drafts built from her own course instead of writing them from scratch. |
| Acceptable waiting time | A draft must arrive faster than she could start writing it herself. The project targets are a first token in under 3 s and a short answer in under 15 s ([model benchmarks](model-benchmarks.md#quality-review-protocol)). |

### Why it must run locally

- **Pupil data.** Marks and difficulties of minors are personal data. With a local Ollama model, nothing leaves the laptop by default: `features.network_enabled` is `false` in `config.json`, and when it is turned on, every outgoing request is shown and must be approved (story [T3-02](user-stories.md#t3--web-search)).
- **No account, no subscription, no network needed.** Preparation works the same at home and in the staff room.
- **Her course is the reference.** SensAI answers from the documents she indexes and names the source file under each grounded answer (story [R1-01](user-stories.md#r1--basic-rag)).

### What the scenario does not need

The scenario also delimits the scope. These exclusions are argued in [technical decisions](decisions.md#adr-07-features-deliberately-not-implemented).

- **Running code written by the model** (catalogue T2). Camille computes averages and statistics, not programs; the custom MCP server gives her exact, tested calculations instead.
- **A public share link** (part of catalogue X3). Her material is printed or sent as a file; a public link would contradict the local-only promise.
- **A pupil-facing assistant.** SensAI is a preparation tool for the teacher; pupils never use it.

## Usage scenario: preparing Monday's fractions lesson

Sunday evening, at home. On Monday, Camille introduces equivalent fractions to one of her *sixième* classes. Each step names the feature it relies on and the story that justifies it. Use only fictional data when replaying it: never real pupil names or marks.

| # | What Camille does | Command or input | Feature | Story | Live demo |
| --- | --- | --- | --- | --- | --- |
| 1 | Starts SensAI with her profile and the mathematics assistant. | `python3 source/main.py --profile demo/profile_camille.json`, `/persona use mathematics-assistant` | M1, A5 | M1-02, A5-01 | 0:00–0:25 |
| 2 | Indexes the progression of the 6e B, the test instructions and the equivalent-fractions lesson. | `/ingest --multiple courses/progression_6e_B.md courses/consignes_evaluation.md courses/fractions_equivalentes.md`, `/rag on` | R1 | R1-01 | 0:25 |
| 3 | Asks when and where the common test takes place. | `Quand a lieu le contrôle commun de fractions et dans quelle salle ?` | R1 | R1-01 | 0:45 |
| 4 | Checks what was actually sent to the model. | `/debug prompt` | EV3 | EV3-02 | 1:05 |
| 5 | Asks something her course does not cover and gets "not found", without a model call. | `Quelle est la capitale de l'Australie ?` | R2 | R2-02 | 1:30 |
| 6 | Imports a colleague's file that hides an instruction for the AI; SensAI warns her. | `/ingest demo/progression_piegee.md`, then a question on it | R1 (safety) | R1-03 | 1:45–2:00 |
| 7 | Computes a pupil's weighted average with the teacher tools server. | `Calcule la moyenne pondérée d'Inès : 16, 14 et 15 avec les coefficients 1, 1 et 2.` | T1 | T1-01 | 2:20 |
| 8 | Sees how fast her laptop answers. | `/stats` | EV3 | EV3-02 | 2:40 |
| 9 | Saves, stops, and resumes the conversation the next day. | `/save`, `exit`, `python3 source/main.py --session <id> --profile demo/profile_camille.json` | M1 | M1-01 | 2:55 |
| 10 | Reuses the earlier answer. | `Transforme ta réponse sur la date et la salle du contrôle commun en un rappel court à écrire au tableau pour les élèves.` | M1 | M1-01 | 3:15 |
| 11 | Checks the context budget and the local metrics. | `/tokens`, `/metrics` | M2, EV3 | M2-01, EV3-01 | 3:35 |
| 12 | Erases the conversation. | `/forget` | M1 (privacy) | M1-03 | 4:00 |
| 13 | Builds the lesson plan, then exports it. | `/artifact init Séance fractions équivalentes`, `/artifact export exports/plan-fractions.md` | M4, X3 | M4-01, X3-02 | No |
| 14 | Generates the 5-question check and saves a pupil copy. | `/quiz --questions 5 --save exports/fractions-eleves.txt --no-answers Les fractions équivalentes` | T5, X3 | T5-01, X3-01 | No |
| 15 | With network access turned on, looks up a source and approves the exact query. | `Cherche sur le web ...`, then `y` | T3 | T3-01, T3-02 | No |
| 16 | A primary-school colleague does the same from a web page. | `python3 source/main.py --web` | X1 | X1-01 | No (recording) |

The [live demonstration](../demo/demo-script.md) lasts about 5 minutes and covers steps 1 to 12. Steps 13 to 16 are covered by automated tests (listed in each story) and by the recordings kept as a fallback; they can be shown on request. All data comes from the fictional course in [`courses/`](../courses/) and [`demo/`](../demo/).

## Hardware feasibility

Camille prepares lessons; she does not use SensAI live in front of pupils. The question is therefore whether a draft arrives faster than she could start writing it, on a laptop a teacher can own.

### Measured machines

| Machine | Chat model | Generation speed | Memory used | Source |
| --- | --- | --- | --- | --- |
| Apple M5, 16 GB unified memory (Camille's reference machine) | `qwen2.5:3b` | 58.1 to 58.7 tokens/s (three runs) | 2.2 GB for the chat model at `num_ctx` 4,096, plus 336 MB for `nomic-embed-text` (274 MB on disk) | [Model benchmarks: Apple M5](model-benchmarks.md#second-machine-apple-m5-unified-memory), 2026-11-01; embedding model size from Ollama `/api/ps` |
| Laptop with AMD Ryzen 7 8845HS, 14 GiB RAM, NVIDIA RTX 4060 (8 GB VRAM) | `qwen2.5:3b` | 111.0 tokens/s (warm mean) | 2,444 MiB peak VRAM (warm) | [Model benchmarks](model-benchmarks.md#results-table---measured-on-2026-10-27), 2026-10-27 |
| Same laptop | `qwen2.5:7b` | 53.7 tokens/s (warm mean) | 4,858 MiB peak VRAM (warm) | [Model benchmarks](model-benchmarks.md#results-table---measured-on-2026-10-27) |
| Laptop without a GPU (CPU only) | - | **Not measured** | **Not measured** | - |

### What it means for Camille's tasks

| Task | Time on the Apple M5 | Basis |
| --- | --- | --- |
| Short answer (two-sentence explanation, check question) | 1.5 to 10 s | Team measurement, 2026-11-01 |
| 15-minute starter activity (517 to 622 output tokens) | 9.4 to 10.7 s in total, almost all of it generation | [Measured](model-benchmarks.md#second-machine-apple-m5-unified-memory) |
| Reading a 2,000-token prompt (persona, course excerpts, history) | 1.35 to 1.47 s | [Measured](model-benchmarks.md#second-machine-apple-m5-unified-memory) (prefill, 2,050-token prompt) |
| 5-question quiz | One generation per attempt, up to 3 attempts (`DEFAULT_MAX_ATTEMPTS` in `tools/structured_output.py`) | **Not measured**; if a quiz is about the size of an activity, about 10 s per attempt and about 30 s in the worst case (estimate) |
| Vector search over 1,000 indexed chunks | ≈ 0.2 s (query embedding call not included) | Measured on synthetic vectors, see [decisions](decisions.md#adr-02-vector-store) |
| Model memory next to her usual applications | ≈ 2.5 GB out of 16 GB | Measured |

At the assumed volume (one plan, four activities, and one quiz per chapter), generation time stays in the order of a minute per chapter. Her review of the drafts, not the model, is the limiting step. A larger model is not justified by these numbers: on the RTX 4060 laptop, `qwen2.5:7b` halves generation speed and doubles memory for a quality gain that has not been demonstrated ([decision ADR-01](decisions.md#adr-01-chat-model)).

### Known risks on other hardware

- **CPU-only laptops are not measured.** Generation would be slower by an unknown factor.
- **30-second request timeout.** On `main` at the time of writing, `utilities/ollama_client.py` sets `DEFAULT_REQUEST_TIMEOUT_SECONDS = 30`. Quiz generation and compression use a non-streamed request, so on a machine where a full quiz takes more than 30 s, the request would fail. This is inferred from the code, not observed.
- **8 GB machines.** The 3B model peaked at 2,444 MiB of VRAM on the RTX 4060 laptop, which suggests it fits; a full SensAI session on an 8 GB machine has not been measured.

## Scenario history

The teacher scenario was **not** fixed at the bootstrap. It was chosen four weeks in, and has stayed the same since. The dates below come from `git log origin/main` and the GitHub tickets.

| Date | Event | Evidence |
| --- | --- | --- |
| 2026-09-23 | Repository bootstrap. No product scenario: the README only says "Mirror of the Sensai project." | Commit `4dd7821` |
| 2026-09-28 to 10-12 | Base loop: Ollama HTTP client, then the CLI. | PR #4, PR #7 |
| 2026-10-12 to 10-13 | Feature epics planned by catalogue area (RAG, Context & Memory, MCP, Tools), before any persona. | Issues #8 to #43 |
| 2026-10-18 | RAG pipeline merged, including code-aware chunking and source-code formats (`.py`, `.c`, `.rs`...). These were designed before the teacher scenario and are not used by it. | PR #48 to #74 |
| 2026-10-22 | **Teacher scenario chosen**: configurable teacher persona. The same day, sandboxed code execution is marked out of scope "for the teacher scenario". | Issue #88 (merged 10-23); issues #36 and #37 edited 10-22 |
| 2026-10-24 to 10-26 | Built for the teacher: `teacher` MCP server (grades, class statistics, timers), `/quiz`, quiz export with a pupil copy, `courses/` and `exports/` permissions. | PR #102, #105, #109, #110, #117 |
| 2026-10-27 | Scenario written down: primary and middle-school teachers; benchmark prompt set `teacher-scenario-v1`. The first version of this document used a teacher named Lucas. | Issues #124, #125, #126; commit `407dc80` |
| 2026-10-30 | Lesson-plan artifact and web search behind a network switch. | PR #119 to #123, #138 |
| 2026-11-01 | Local web page for teachers who prefer not to use a terminal; demo script with Camille, a *sixième* mathematics teacher. | PR #140, #141 |
| 2026-11-01 | This revision: the persona of this document becomes Camille, so that the document and the demo describe the same teacher and the same lesson. | This document |

Since 2026-10-22, the audience (primary and middle-school teachers), the subject of the demo (fractions in *sixième*), and the reason for local execution (pupil data) have not changed. Only the persona's name changed, from Lucas to Camille. Every feature merged after that date was built for this scenario. The RAG pipeline is the exception: it predates the scenario, which is why it still contains code-oriented chunking that the teacher does not use (see [decisions](decisions.md#4-write-the-scenario-and-an-evaluation-set-first)).

## User stories

User stories, acceptance criteria, and the traceability tables have moved to **[docs/user-stories.md](user-stories.md)**. The 14 stories of the first version are rewritten there; the [former identifiers](user-stories.md#former-story-identifiers) table maps them to the new ones.

## Limits and future compatibility

- Generated explanations, quiz answer keys, and grades can be wrong. The quiz schema guarantees the structure of a quiz, not the correctness of its answers; the teacher reviews everything before use.
- SensAI exposes the same features to every teacher, including high-school teachers. The prompts, the chosen local model, and this scenario are designed and checked for primary and middle school, where the expected complexity is lower. Future high-school work concerns model evaluation, subject-specific prompts, curriculum documents, and validation with high-school teachers; it does not mean that a feature is unavailable.
- Persistent sessions preserve conversation context; they are not a long-term record of each pupil.
