# SensAI live demonstration

**Target duration:** 4 minutes 30 seconds
**Audience:** primary and middle-school teachers

## Roles

| Presenter | Responsibility |
| --- | --- |
| Presenter 1 | Profile, persona, save, restart, conclusion |
| Presenter 2 | RAG ingestion and grounded preparation |
| Presenter 3 | Resumed conversation, tool call, token and monitoring views |

One or two presenters may also use this script.

## Before the timer starts

In terminal A, start Ollama:

```bash
ollama serve
```

If the terminal reports that the address is already in use, Ollama is already running. Do not start a second server.

In terminal B, from the repository root:

```bash
source .venv/bin/activate
ollama list
python3 source/main.py --profile demo/profile_camille.json
```

## Live script

| Time | Presenter | Action | Exact command or input | Expected proof | Spoken line |
| --- | --- | --- | --- | --- | --- |
| 0:00-0:20 | 1 | Explain the profile and persona distinction. | `/persona status` | Active persona metadata is displayed. | “The profile describes Camille and her teaching context. The persona describes the role and tone of the assistant.” |
| 0:20-0:35 | 1 | Select the mathematics persona. | `/persona use mathematics-assistant` | Persona change confirmation. | “We select a mathematics assistant without changing the application code.” |
| 0:35-0:55 | 2 | Index the course and enable RAG. | `/ingest demo/fractions_6e.md` then `/rag on` | Successful indexing and RAG mode confirmation. | “The course is stored locally and can now ground answers.” |
| 0:55-1:20 | 2 | Ask for a classroom-ready explanation. | `Prépare une explication de deux phrases sur les fractions équivalentes pour une classe de sixième.` | A short answer and `Sources: fractions_6e.md ...`. | “The answer uses the indexed support and remains traceable to its source.” |
| 1:20-1:35 | 2 | Save the planning session. | `/save` | `Session saved at sessions/<SESSION_ID>.json` | “The conversation is saved locally. We now note the identifier shown on screen.” |
| 1:35-1:50 | 1 | Record the identifier and exit. | `exit` | One save confirmation, then the shell prompt. | “The session is saved once on exit. We will now resume it as if Camille returned later.” |
| 1:50-2:15 | 1 | Restart with the displayed identifier. | `python3 source/main.py --session <SESSION_ID> --profile demo/profile_camille.json` | The `You:` prompt opens. | “Replace `<SESSION_ID>` with the identifier displayed by `/save`, without the `.json` extension.” |
| 2:15-2:45 | 3 | Refer to the earlier answer. | `Peux-tu transformer l'explication précédente en une question de vérification ?` | The model understands what “previous explanation” means. | “The user and assistant history was restored and is sent to the model again.” |
| 2:45-3:00 | 3 | Show token tracking. | `/tokens` | Last-turn, context-window, and session token information. | “This exposes context usage, which matters when a conversation becomes long.” |
| 3:00-3:20 | 3 | Demonstrate the local time tool. | `/tools` then `Quelle heure est-il ?` | Tool list and a `[tool] get_time({})` call. | “Tools are explicit and validated. This local time comes from the machine, not from model memory.” |
| 3:20-3:40 | 3 | Show observability. | `/metrics` | Local dashboard with call count, latency, token estimates, RAG, and tools. | “The dashboard lets developers inspect technical behaviour without storing raw conversation content by default.” |
| 3:40-4:10 | 1 | State the privacy trade-off. | No command. | - | “Sessions and profiles remain local, which limits cross-teacher exposure. However, they may contain personal information and must be protected from a compromised device or accidental Git commit.” |
| 4:10-4:30 | 1 | State the product limitation. | No command. | - | “Persistent sessions preserve conversation context; they are not an automatic long-term learner-memory system, and generated teaching material still needs teacher review.” |

## Acceptable variations

- The generated explanation and verification question can vary.
- The displayed session identifier is random and must be copied exactly.
- If the session has already been saved, a later `exit` updates the same session file.
- `/tokens` values depend on the local model response and context.

## Fallback plan

If the restart or session loading fails, keep the demonstration moving:

```bash
python3 -m unittest tests.test_cli.TestCLICommandDispatch tests.test_cli.TestArtifactSessionLifecycle
```

This automated check verifies the save/exit dispatch path and persistence lifecycle. Explain that the team documents a failed session load as a controlled user-facing error rather than exposing a traceback.

## Rehearsal checklist

- [ ] Complete the core demonstration prerequisites.
- [ ] Confirm that `features.memory_enabled` is `true` in `config.json`.
- [ ] Run `/save` once and practise copying the generated session identifier.
- [ ] Restart using `--session <SESSION_ID>` before the presentation.
- [ ] Verify that a question referring to the “previous explanation” works after restart.
- [ ] Verify `/tokens`, `/tools`, and `/metrics` on the demonstration machine.
- [ ] Keep a screenshot of the saved-session output as backup.

