# Cost-benefit analysis

[Home](../README.md) · [Product scenario](products-scenario.md) · [Local data and GDPR](RGPD.md) · [Model benchmarks](model-benchmarks.md)

This analysis asks whether SensAI is worth installing for a teacher, and worth maintaining
for the team. It follows the demonstration persona: **Camille**, a mathematics teacher
preparing a fractions sequence for a *sixième* (Year 7) class on her own laptop.

Every value is labelled:

- **Measured**: observed on our machines, with the date and source;
- **Computed**: derived from other values, formula shown;
- **Audit**: from the team's defense audit run (not committed). The RAG A/B figure now comes
  from the reproducible evaluation script in `docs/evaluation.md`;
- **Reported**: published by a third party, with a link and the consultation date;
- **Assumption**: our estimate, with its reasoning. Assumptions are not results.

"M5" is an Apple M5 laptop with 16 GB unified memory (Ollama 0.40.7) and "RTX 4060" is the
Linux laptop with an 8 GB RTX 4060 Laptop GPU (Ollama 0.24.0) described in
[model-benchmarks.md](model-benchmarks.md).

## Verdict

SensAI is worth using today for **one task family**: drafting lesson material (activities,
explanations, quiz drafts) from the teacher's own course documents, on a laptop that
already has a recent GPU or Apple silicon, with every output reviewed before class. It
saves drafting time and keeps documents on the device. It does **not** save checking time:
a 3B local model makes arithmetic and answer-key errors that a teacher must catch.

**Recommended deployment scope: primary and middle-school teachers**, individually, on a
single-user device that the school has approved and encrypted, for preparation work with
public, fictional, or anonymised content. Not recommended: pupil-facing use, automated
grading, real pupil data without the head of school's approval, shared classroom computers,
or high-school content, which the product has not been evaluated for
([product scenario](products-scenario.md#target-users)).

## 1. Benefits for the teacher

| Benefit | Evidence | Label |
| --- | --- | --- |
| A 15-minute activity drafted in seconds | 517–866 output tokens in 9.4–15.0 s with `qwen2.5:3b` on the M5, 2026-11-01 (the 866-token draft used the mathematics persona) | Measured |
| Short answers without waiting | 0.3–4.2 s for 12–239 output tokens with the mathematics persona, M5, warm; 1.7 s for the first request, including 0.55 s of model loading | Measured |
| Answers grounded in her own course | Course-question accuracy 0% without RAG, 83% with RAG (12 questions × 3 runs: 0/36 against 30/36); 12/12 "not found" on facts absent from the course, no invented answer | Measured, [evaluation](evaluation.md) |
| Course documents and conversations stay on the laptop | `features.network_enabled` is `false` by default; network tools ask a `[y/N]` confirmation; monitoring stores no prompt or answer ([RGPD.md](RGPD.md)) | Code and configuration |
| Deterministic classroom calculations | Weighted grade, class statistics, and timer run as Python code in the teacher MCP server (`source/mcp/servers/teacher_tools.py`), not as model arithmetic | Code |
| No subscription and no account | No recurring licence fee; no account creation, which the CNIL advises against for pupils on consumer tools ([CNIL FAQ for teachers](https://www.cnil.fr/fr/enseignant-usage-systeme-ia), 20 June 2025) | Reported |
| Works offline | After the one-time model download, every model call goes to `http://localhost:11434` | Configuration |
| Continuity between evenings | `/save`, `/load`, profile, and context compression ([sessions and profile](features/sessions-and-profile.md)) | Code |

## 2. Costs for the teacher

| Cost | Value | Label |
| --- | --- | --- |
| Compatible hardware | Works on the M5 (16 GB) and the RTX 4060 (8 GB VRAM). CPU-only laptops were **not measured**. | Measured |
| Speed on a CPU-only laptop | Upper bound only: a dual-channel DDR4-3200 laptop has 2 × 8 bytes × 3,200 MT/s = 51.2 GB/s, so at most 51.2 / 1.93 ≈ 26 tok/s for `qwen2.5:3b`, and a 600-token activity in 23 s or more. Real CPU speed is usually lower than this ceiling. | Computed ceiling, not measured |
| Hardware purchase | Not estimated: we have no sourced price for a school-issued laptop. SensAI only avoids a subscription if the teacher's existing device is compatible. | Not estimated |
| Memory while in use | 2.2 GB (`qwen2.5:3b`) + 335 MB (`nomic-embed-text`) according to `ollama ps` on the M5, about 2.5 GB of 16 GB; 2,444 MiB of the RTX 4060's 8,188 MiB. Ollama unloads a model a few minutes after the last request (`UNTIL` column of `ollama ps`). | Measured |
| Disk | 4.22 GB of models: 1.93 GB `qwen2.5:3b` + 2.02 GB `llama3.2:latest` fallback + 0.27 GB `nomic-embed-text` (Ollama manifests). Ollama and Python dependencies: not measured. | Measured |
| First download | 4.22 GB × 8 / 50 Mbit/s ≈ 11 min at an assumed 50 Mbit/s connection (28 min at 20 Mbit/s). | Computed from an assumed bandwidth |
| Installation | Python 3.11+, Ollama, three `ollama pull` commands, then `python3 source/main.py --web` ([README](../README.md#quick-start)). Installation time was not measured. | Documented |
| **Checking every output** | Quiz answer keys wrong in 14 of 18 generated questions; fraction ordering correct in 1 of 3 attempts; hallucinated answers to 5 of 15 questions about facts absent from the model's sources, without RAG. | Audit |
| | "Which fraction is larger, 3/4 or 5/8?" answered "5/8" in 2 of 2 runs, with and without the mathematics persona (M5, 2026-11-01). | Measured |
| Energy | Not measured. Possible method: `sudo powermetrics --samplers gpu_power` on macOS, `nvidia-smi --query-gpu=power.draw --format=csv` on NVIDIA. | Not measured |

### Teacher time: an explicit assumption

We have **not measured** time saved. The table below only shows how the measured
generation time and the audit error rates constrain any estimate.

| Item | Range | Reasoning | Label |
| --- | --- | --- | --- |
| Writing a 15-minute activity by hand (A) | 20–40 min | Choose exercises, write instructions, write a correction | Assumption |
| Generating the draft with SensAI (G) | under 0.3 min | 9.4–15.0 s on the M5 | Measured |
| Reviewing and correcting the draft (R) | 10–20 min | Every example and result must be checked, given the audit error rates | Assumption |
| Net saving per activity: A − G − R | about 0–30 min | (20 − 0.3 − 20) to (40 − 0.3 − 10), rounded | Computed from the assumptions |
| Quiz with answer key | about 0, possibly negative | With 14 of 18 keys wrong, every key must be re-solved; only the drafting of question wording is saved | Assumption |

The range includes zero, so the honest claim is: **SensAI can save drafting time on
activities and explanations; it saves little or nothing on answer keys until they are
checked automatically.** A timed comparison (the same teachers preparing the same
activity with and without SensAI) is required before any time-saving figure is quoted.

## 3. Comparison with a cloud assistant

Prices **reported** on 2026-11-01, as listed by each provider (in US dollars):

| | SensAI (local) | ChatGPT Plus | Mistral Le Chat Pro |
| --- | --- | --- | --- |
| Price per teacher | 0 recurring software cost | USD 20 per month ([OpenAI Help Center](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus)) | USD 14.99 per month excluding tax ([Mistral pricing](https://mistral.ai/pricing)) |
| Per year (computed: × 12) | 0 | USD 240 | USD 179.88 |
| Hardware | Recent GPU or Apple silicon (tested) | Any device with a browser | Any device with a browser |
| Non-public course documents | Stay on the device | Uploaded to the provider | Uploaded to the provider |
| Pupil personal data under the French framework | Not a consumer service: possible only with the head of school's agreement and data minimisation | Forbidden: consumer AI may only receive data that could be made public | Same as ChatGPT Plus |
| Offline use | Yes, after the model download | No | No |
| Answer quality on our tasks | Measured errors (section 2) | Not measured by us | Not measured by us |

The OpenAI price page could not be retrieved automatically from our network; the USD 20
figure comes from OpenAI's Help Center page as indexed by a web search on 2026-11-01. Check
both pages before quoting these prices.

The French Ministry of Education's framework (*L'IA en éducation : cadre d'usage*, June
2025, [official page](https://www.education.gouv.fr/cadre-d-usage-de-l-ia-en-education-450647);
quotes taken from the [PDF copy hosted by the Orléans-Tours academy](https://pedagogie.ac-orleans-tours.fr/documents/pdf/ia_en_education_-_cadre_usage.pdf),
because the official page could not be retrieved automatically) sets these rules:

- consumer AI services are allowed "sous réserve qu'aucune donnée confidentielle ou à
  caractère personnel ne soit utilisée": only data that could be made public (official
  curricula, open resources, anonymised statistics);
- any use respects the GDPR, "notamment le principe de minimisation et l'accord du
  responsable de traitement" (the head of school);
- no staff member may ask pupils to use a consumer AI service that requires a personal
  account.

**Honest reading:** Camille's fractions activity built from the official curriculum
contains no personal data, so a cloud assistant is allowed for it and runs larger models.
SensAI's advantage is real for **non-public course material, offline work, and zero
recurring cost**, and becomes decisive only if pupil data is involved, which still
requires the school's approval and a protected device.

## 4. Privacy: local benefits against device risks

| Local benefit | Device risk that remains | What reduces it |
| --- | --- | --- |
| No prompt, document, or answer is sent to a third party by default | Files at rest are **not encrypted**: `sessions/`, `data/sensai_rag.sqlite3`, `logs/sensai.log`, `profile.json` | Full-disk encryption (FileVault, BitLocker) managed by the school; `/forget`, `/rag reset`, `scripts/clean_sessions.sh` |
| Session files and profiles get `0700`/`0600` permissions | Permissions protect against other OS accounts, not against anyone using the same session | One OS account per teacher, screen lock; no shared classroom computer |
| The web interface listens on `127.0.0.1` only, checks the `Host` header, and refuses cross-site requests (`source/web/server.py`) | It has **no login**: any user or process on the same machine can open `http://localhost:8000` and read conversations | Single-user device; stop the server (`Ctrl+C`) after use |
| Network tools are off by default and ask `[y/N]` before each call | The confirmation cannot detect a pupil's name in the request | Keep `network_enabled` false for pupil-related work ([RGPD.md](RGPD.md)) |
| A malicious document only affects the computer where it was indexed | That document can still mislead the model there (prompt injection); configured MCP servers run local commands | Index trusted documents only; review MCP servers and `permissions.json` |
| Stolen laptop: no cloud account to compromise | Everything stored locally is readable without disk encryption | Encryption, incident procedure in [RGPD.md](RGPD.md#what-happens-in-case-of-a-leak) |

Local processing removes the cloud provider from the data flow; it does not make the device
secure. The CNIL's FAQ for teachers also asks that the head of school be informed and the
data protection officer involved before a tool processes pupil data
([CNIL, 20 June 2025](https://www.cnil.fr/fr/enseignant-usage-systeme-ia)).

## 5. Costs and benefits for the development team

| Item | Evidence on `main`, 2026-11-01 | Ongoing cost |
| --- | --- | --- |
| Automated tests | 592 test methods (`git grep -c 'def test_'` over `tests/`); 8,323 lines of tests for 8,123 lines of source code. The Ollama client and network are mocked, so the suite runs offline. | Every feature change needs tests; mocks do not catch real-model behaviour (wrong argument types, wrong answer keys), so live checks stay manual |
| Benchmarks | Two machines, two Ollama versions (0.24.0 and 0.40.7); one full comparison is 3 models × 4 prompt cases × 3 runs = 36 requests, plus 12 responses per model to review by hand | Re-run after every model, prompt, or Ollama change; the manual quality review of two models is still pending; raw results are Git-ignored, so values must be copied into the docs |
| Documentation | 1,437 lines of Markdown in `docs/` before the defense documents | Keep commands, defaults, and numbers in sync with the code |
| Security and privacy | ZIP limits, permissions, network confirmation, local-only web server ([RGPD.md](RGPD.md)) | Prompt injection through documents and MCP outputs, dependency updates (`requirements.txt`), MCP server review |
| Model licence | `qwen2.5:3b` is under the Qwen Research License ("research or evaluation purposes only") | A school deployment needs a licence review or a switch to an Apache 2.0 model (`qwen2.5:7b` measured, `qwen3:4b` not measured), then a new benchmark |
| Support | No remote telemetry by design | Teachers must report problems themselves; the expected first issues are Ollama not running (the web page shows a banner), a missing model (automatic fallback), and hardware that is too weak (unmeasured) |
| Benefits | No API bill, no per-request cost, no server to operate, no data-processing contract with an AI provider | Hardware heterogeneity becomes the team's problem instead of the provider's |

## 6. Main limitations and trade-offs

1. **Correctness against speed and size.** A 3B model answers in seconds on a laptop but
   makes maths and answer-key errors (section 2). A 7B would likely make fewer
   (Qwen reports MATH 75.5 against 65.9, per the Qwen2.5 model card)
   at half the speed; we have not measured that gain on our tasks.
2. **Local privacy against device security.** Data stays on the laptop, but unencrypted
   and behind no login. The privacy benefit depends on the school's device management.
3. **No subscription against hardware requirements.** Zero recurring cost only holds for
   teachers who already own a compatible device; CPU-only performance is unknown.
4. **French course material against an English-centric embedding model.** Retrieval
   separates topics well but ranks passages of the same topic poorly in our French example; no reranking step
   exists.
5. **Small context against memory use.** `num_ctx` 4096 keeps the KV cache at 144 MiB, but
   long preparation sessions need compression, which costs an extra model call and can
   lose detail.
6. **Free model against licence terms.** The fastest valid tool caller is under a
   non-commercial research licence.

## 7. Decision table

| Feature or choice | Benefit | Risk | Mitigation in SensAI | Remaining trade-off |
| --- | --- | --- | --- | --- |
| Local inference with Ollama | No third-party processing, offline, no subscription | Smaller model, hardware requirement | 3B Q4_K_M model (2.2 GB loaded), automatic fallback model | Quality below large cloud models; CPU-only unmeasured |
| RAG on the teacher's documents | 0% to 83% course-question accuracy ([evaluation](evaluation.md)) | Wrong passage retrieved; instructions hidden in a document | Source display, name and category filters, `/rag reset`, trusted documents only | No reranking, no OCR, English-centric embeddings |
| Quiz generation | A structured draft in seconds | Wrong answer keys (14 of 18 in the audit) | Structure validation, learner copy without answers, warning in the README | Every key must be checked by the teacher |
| Teacher MCP tools (grade, statistics, timer) | Calculations done by code, not by the model | The model passes wrong arguments; marks are personal data | Argument validation in `teacher_tools.py`; fictional or anonymised marks | The teacher checks the inputs the model chose |
| Sessions and profile | Resume preparation; answers in the teacher's language | Unencrypted files that may contain personal data | `0600`/`0700` permissions, `/forget`, profile limited to teaching preferences | Disk encryption is the school's responsibility |
| Local web interface | No terminal needed | No login on `localhost:8000` | Listens on `127.0.0.1` only, `Host` check, no cross-site requests | Assumes one user per device |
| Network tools (web fetch MCP) | Access to public online resources | Data leaves the device | Off by default, `[y/N]` confirmation per call | The confirmation cannot recognise personal data |
| Default model `qwen2.5:3b` | Valid tool calls 3/3, fastest warm latency | Research licence, maths errors | Model name in `config.json`; Apache 2.0 alternatives identified | A switch requires a new benchmark |
| `num_ctx` 4096 | 144 MiB KV cache; fits persona, 4,000 characters of RAG, and a full activity | Long sessions exceed the window | `/tokens`, warning at 80%, compression at 85% | Compression costs a model call and may drop detail |

## 8. Realistic workflow today and future work

What Camille can realistically do **today** (the [demonstration script](../demo/demo-script.md)
shows the indexing, grounded-explanation, and save-and-resume steps):

| Step | What SensAI does | What Camille still checks |
| --- | --- | --- |
| Index her fractions course (`/ingest`, or "My course documents" in the web page) and turn on RAG | Splits, embeds, and stores it locally | That the file is her own or public material |
| Ask a question about her course | Answers from the retrieved excerpts and shows the sources | The cited excerpt really supports the answer |
| Draft a 15-minute activity on equivalent fractions | A complete draft in about 10–15 s (M5) | Every example, result, and fraction comparison |
| Draft a short quiz | Questions, options, and a proposed key | **Every answer key**, re-solved by hand |
| Compute a weighted average for fictional or anonymised marks | Exact result from the teacher tool | The marks and coefficients the model passed to the tool |
| Save the session and resume the next evening | Restores history and profile | Nothing, apart from deleting old sessions |

**Future work**, not realistic today:

- distributing quizzes without checking their keys: needs automatic verification of
  numeric answers (for example exact fraction arithmetic with Python's `fractions` module)
  and a measured error rate after that change;
- using real pupil data (grades, difficulties): needs the head of school's approval,
  encryption at rest, and a login on the web interface;
- one SensAI server for a whole school: needs authentication, per-teacher isolation, and
  encrypted storage;
- pupil-facing use or automated grading: out of scope. The French framework allows pupils
  to use generative AI in class only from *4e*, under the teacher's supervision, so not in
  Camille's *sixième* class; and AI systems "intended to be used to
  evaluate learning outcomes" are high-risk under Annex III, point 3(b) of the EU AI Act
  ([Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj),
  [readable Annex III](https://artificialintelligenceact.eu/annex/3/));
- CPU-only school laptops: measure before recommending;
- high-school content: evaluate with harder material, probably with a 7B model;
- a measured teacher time saving (the RAG accuracy is now measured: 0% → 83%,
  [evaluation](evaluation.md)).
