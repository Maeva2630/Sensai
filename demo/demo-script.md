# SensAI live demonstration

**Scene:** Sunday evening. Camille teaches mathematics to the 6e B of a *collège*. She
prepares the fractions sequence of the week with her own course documents, on her own
laptop, without sending pupil information to an online service.

**Target duration:** 5 minutes · **Presenters:** 3 · **Machine:** the presenter's laptop
(Apple M5, 16 GB, qwen2.5:3b ≈ 59 tokens/s), Ollama already running.

All data is fictional (`courses/`, `demo/`). Persona: [product scenario](../docs/products-scenario.md).

## Roles

| Presenter | Plays | Shows |
| --- | --- | --- |
| P1 | Narrator and Camille's colleague | Scene, session restore, metrics, privacy, conclusion |
| P2 | Camille | Her course, grounded answers, the prompt actually sent, the "not found" guard |
| P3 | The school's IT referent | Poisoned document, MCP grade tool, speed, limits |

## Before the timer (checklist, 5 minutes before)

```bash
ollama serve                      # terminal A (skip if "address already in use")
source .venv/bin/activate         # terminal B, repository root
pip install -r requirements.txt   # once: PDF/DOCX support
ollama list                       # qwen2.5:3b, llama3.2:latest, nomic-embed-text
python3 tests/test_ollama_client.py --integration   # expect "VERDICT: PASS (4/4 steps)"
scripts/clean_sessions.sh         # no conversation left from rehearsal
rm -f data/sensai_rag.sqlite3     # empty document index (or /rag reset inside SensAI)
ollama run qwen2.5:3b "OK" >/dev/null   # load the model in memory: no cold start on stage
```

- [ ] `config.json`: `memory_enabled` true, `monitoring.enabled` true, `network_enabled` false.
- [ ] Terminal font large enough for the back row; `NO_COLOR` unset.
- [ ] Terminal C prepared with `python3 source/main.py --profile demo/profile_camille.json --session ` (the identifier is pasted at 2:55).
- [ ] Fallback recordings open in a background window (see the end of this document).

## Live script

Start: `python3 source/main.py --profile demo/profile_camille.json`

| Time | Who | Action (type exactly) | Expected proof on screen | Spoken line |
| --- | --- | --- | --- | --- |
| 0:00 | P1 | - | - | "Camille has 26 pupils in 6e B and three hours on Sunday to prepare the fractions week. Her documents and her pupils' marks must stay on her laptop. SensAI is her assistant, running entirely locally." |
| 0:15 | P2 | `/persona use mathematics-assistant` | `Active persona changed to: Mathematics assistant` | "I choose the mathematics assistant; my profile already says French, sixth grade, short classroom activities." |
| 0:25 | P2 | `/ingest --multiple courses/progression_6e_B.md courses/consignes_evaluation.md courses/fractions_equivalentes.md` then `/rag on` | Three `Document indexed` lines, `RAG mode enabled` | "I index my own course: it stays in a local database." |
| 0:45 | P2 | `Quand a lieu le contrôle commun de fractions et dans quelle salle ?` | `jeudi 17 décembre 2026 à 10h30, salle B12` + `Sources: progression_6e_B.md` | "The answer comes from my document, and SensAI, not the model, names the source." |
| 1:05 | P2 | `/debug prompt` | `Settings: … temperature=0.2, num_ctx=4096`, `RAG rules`, `User message with document excerpts (framed as data)` | "This is exactly what was sent: our rules as system message, my document as data next to my question, and the settings." |
| 1:30 | P2 | `Quelle est la capitale de l'Australie ?` | `Je ne trouve pas cette information dans les documents indexés.` + `the model was not asked` | "No relevant passage: SensAI does not even ask the model, so it cannot invent an answer." |
| 1:45 | P3 | `/ingest demo/progression_piegee.md` | `Warning: progression_piegee.md contains text that looks like instructions to an AI assistant` + the `SYSTEM:` line | "A colleague shared this file. It hides an instruction for the AI. SensAI flags it at import." |
| 2:00 | P3 | `Que prévoit la séance 2 de la progression ?` | Answer about fractions équivalentes + `Warning: an excerpt used for this answer (progression_piegee.md) …` | "We measured it: a 3B model still obeys such text about one time in four. So every write and every network call needs a human yes, and Camille is told which document to check." |
| 2:20 | P3 | `Calcule la moyenne pondérée d'Inès : 16, 14 et 15 avec les coefficients 1, 1 et 2.` | `[tool] teacher__calculate_grade(...)` → `Weighted average: 15/20` | "Our own MCP server does the arithmetic: the model chooses the tool, the code computes." |
| 2:40 | P3 | `/stats` | `Generation: … ≈ 59 tokens/s` | "59 tokens per second on this laptop: generation is limited by memory bandwidth, 1.9 GB of weights read for every token. A one-page activity takes about 10 seconds." |
| 2:55 | P1 | `/save`, copy the identifier, `exit`; in terminal C paste it and press Enter | `Session saved at sessions/<id>.json`, then `You:` | "Camille stops and comes back on Monday." |
| 3:15 | P1 | `Transforme ta réponse sur la date et la salle du contrôle commun en un rappel court à écrire au tableau pour les élèves.` | `Rappel : Contrôle commun de fractions le jeudi 17 décembre à 10h30 en salle B12.` (2/2 in rehearsal) | "The conversation was restored: 'the answer about the test' is understood." |
| 3:35 | P1 | `/tokens` then `/metrics` | Context usage in tokens; dashboard with calls, latency, RAG share, tool calls | "We track the context budget and the system's behaviour, without storing any conversation text in the metrics." |
| 4:00 | P1 | `/forget` | `Saved copy of this session deleted.` | "The data stays on this laptop, readable by Camille's account only, and she can erase it in one command." |
| 4:15 | P1 | - | - | "What SensAI does not do: its quiz answer keys can be wrong, about three out of four in our tests, and a hidden instruction can still influence an answer. Camille reviews everything; SensAI makes that review easy and keeps actions under her control." |
| 4:40 | P1 | - | - | "Measured on 12 questions about Camille's course, 3 runs: correct answers go from 0 % without her documents to 83 % with them, and on 4 questions the course does not cover, 12 answers out of 12 say 'not found' instead of inventing." ([evaluation](../docs/evaluation.md); replay: `python3 scripts/evaluate_rag.py --quick`) |

## Acceptable variations

- Wording of answers varies; the facts (17 décembre 2026, 10h30, B12, 15/20) must not.
- `/stats` varies by a few tokens/s; the first answer after a cold start shows a load time.
- If the injected document makes the model write "PAMPLEMOUSSE", **say it**: it is the
  measured limit, and the warning line is the point of the scene.

## If something fails on stage

| Failure | Do this | Say this |
| --- | --- | --- |
| Ollama not answering / model missing | Show recording 1, then run `python3 scripts/run_tests.py` (729 tests, offline) | "The engine is tested without the model; here is the run we recorded this morning." |
| The date answer is wrong | `/debug prompt` to show the retrieved excerpt | "The right passage was retrieved; this is the 3B model's limit, which is why sources are shown." |
| The grade tool is not called | Ask again with "utilise l'outil de moyenne" | "The model decides to call tools; our routing only offers the right ones." |
| Session restore fails | Show recording 2 | "Errors are reported, never a traceback." |
| Time is short | Skip 1:05 and 3:35 | - |

## Jury replay commands (questions after the demo)

```bash
python3 scripts/evaluate_rag.py --quick        # RAG on vs off, about 1 minute, prints VERDICT
python3 scripts/probe_limits.py --quick        # hallucination, codename, long prompt, injection
python3 scripts/run_tests.py                   # per-feature results and percentage
python3 source/main.py --temperature 1.2       # then ask the same question three times
```

Inside SensAI: `/set temperature 0` then `/set temperature 1.2` with the same request shows
the effect of temperature in seconds; `/set context 8192` changes the window.

## Recordings to prepare (fallback)

Record them on the demo laptop the day before (QuickTime or `asciinema`), and keep them in
`demo/recordings/` (not committed):

1. The full live script above, in one take.
2. Session save, exit and restore.
3. `python3 scripts/evaluate_rag.py --quick` with its verdict.
4. The web page (`python3 source/main.py --web`): upload a course, ask the date question,
   generate a quiz, refuse a web search.
