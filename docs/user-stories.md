# User stories and traceability

[Home](../README.md) · [Product scenario](products-scenario.md) · [Technical decisions](decisions.md) · [Demo script](../demo/demo-script.md)

This document is the single reference for SensAI's user stories. It answers three questions:

1. Which catalogue features are implemented, and how completely? ([Feature inventory](#feature-inventory))
2. Which use case justifies each feature? ([User stories](#user-stories), at least two per feature)
3. How does each need of the subject and of the scenario reach code, tests, and the demo? ([Brief to implementation](#brief-to-implementation))

All stories follow the persona of [docs/products-scenario.md](products-scenario.md): **Camille**, a mathematics teacher of *sixième* classes (first year of French *collège*, pupils aged 11–12), preparing a lesson on equivalent fractions on her own laptop. A few stories use the people around her: a colleague, the school IT technician, and the school data protection officer.

## How to read a story

- Every story has an identifier `<catalogue code>-<number>` (for example `R1-01`). The code is the feature the story justifies. When a story relies on a second feature, that feature is named in the `Code:` line, never in the story itself.
- `Code:` lists the main modules under `source/`.
- `Tests:` lists automated tests from `tests/`, written `file.py::Class::test_method`. Run one with `python3 -m unittest tests.<file>.<Class>.<test_method>`. Every cited test exists on `main` and runs offline.
- `Shown:` gives the step of the [live demo script](../demo/demo-script.md) by time slot and action, or `doc only` when the story is not part of the 4 min 30 s demonstration.
- Messages quoted in acceptance criteria are those printed by `main` at commit `6487dd6` (2026-11-01).

> **Duplicate test names.** In `tests/test_cli.py` on `main` (commit `6487dd6`), seven method names are defined more than once in the same class (`TestCLICommandDispatch` and `TestCLIExceptionHandling`). Python keeps only the last definition, so only that one runs. Three of them are cited below (`test_tokens_command_tracks_per_turn_and_cumulative_usage`, `test_slash_load_restores_conversation_context_for_next_turn`, `test_auto_compression_triggered_when_budget_critical`); in each case the cited behaviour is the one checked by the last definition in the file. The duplicates are being removed at the time of writing, and the cited names are kept.

## Feature inventory

Codes and names come from `subjects/features_catalog.pdf`. **Full** means the catalogue description is covered; **partial** means part of it is, and the gap is stated.

| Code | Catalogue feature | Status | Main modules | Commands and configuration | Stories |
| --- | --- | --- | --- | --- | --- |
| B1 | Functional CLI chatbot (base loop) | Full | `cli/cli.py`, `utilities/ollama_client.py`, `cli/error_handling.py`, `cli/fallback.py`, `config.py` | `python3 source/main.py`, `config.json` → `model` | B1-01 to B1-03 |
| R1 | Basic RAG | Full | `rag/ingestion.py`, `rag/chunking.py`, `rag/embeddings.py`, `rag/storage.py`, `rag/context.py` | `/ingest`, `/rag on\|off\|status`, `/rag reset`, `/sources`, `config.json` → `rag` | R1-01 to R1-03 |
| R2 | Advanced RAG | **Partial**: metadata filtering and per-chunk relevance threshold. No reranking (ticket #17 is closed but no reranking code exists, see [decisions](decisions.md#adr-07-features-deliberately-not-implemented)). | `rag/storage.py` (`MetadataFilter`), `rag/context.py` (`_filter_by_relevance`) | `/rag filter name\|category\|clear`, `rag.relevance_threshold` | R2-01, R2-02 |
| M1 | Session persistence and profiles | Full | `memory/session_manager.py`, `memory/profile_manager.py`, `memory/private_files.py`, `cli/commands/memory.py` | `/save`, `/load`, `/forget`, `--session`, `--profile`, `profile.json` | M1-01 to M1-03 |
| M2 | Token budgeting and semantic compression | Full. On `main`, usage is measured after each answer rather than before sending (see [decisions](decisions.md#2-check-the-token-budget-before-sending)). | `cli/session.py`, `memory/compression.py` | `/tokens`, `/compress`, `model.context_window` | M2-01, M2-02 |
| M4 | Artifact and state document | Full | `memory/artifact_manager.py`, `tools/artifact_tool.py`, `cli/commands/artifact.py` | `edit_artifact` tool, `/artifact`, `/doc` | M4-01, M4-02 |
| T1 | MCP | Full: own server (`teacher`) and a preexisting one (`mcp-server-fetch`) | `mcp/client.py`, `mcp/manager.py`, `mcp/config.py`, `mcp/servers/teacher_tools.py`, `cli/tool_calls.py`, `tools/registry.py` | `/tools`, `mcp_servers.json` | T1-01, T1-02 |
| T3 | Web search | Full, off by default | `tools/web_search.py`, `tools/safety.py`, `cli/tool_calls.py` | `features.network_enabled`, y/N confirmation | T3-01, T3-02 |
| T4 | File access within permissions | Full | `tools/permissions.py`, `tools/filesystem.py` | `permissions.json`, `/permissions` | T4-01, T4-02 |
| T5 | Structured output | Full | `tools/structured_output.py`, `cli/commands/tools.py` | `/quiz` | T5-01, T5-02 |
| A5 | Persona | Full | `config.py`, `prompts/*.txt`, `cli/commands/persona.py` | `/persona list\|status\|use`, `config.json` → `personas` | A5-01, A5-02 |
| EV3 | Logging and monitoring dashboard | Full (logs deliberately contain no prompt or response text) | `observability/app_logging.py`, `observability/monitoring.py`, `cli/prompt_inspection.py` | `/metrics`, `/debug prompt`, `monitoring`, `logging` | EV3-01, EV3-02 |
| X1 | Web UI | Full | `web/app.py`, `web/server.py`, `web/static/index.html` | `--web`, `--port` | X1-01 to X1-03 |
| X3 | Export and publication | **Partial**: quiz export (text, JSON, pupil copy) and lesson-plan export (Markdown). No PDF, no conversation export, no public share link (excluded on purpose, see [decisions](decisions.md#adr-07-features-deliberately-not-implemented)). | `tools/quiz_export.py`, `cli/commands/artifact.py`, `web/server.py` | `/quiz --save`, `/artifact export` | X3-01, X3-02 |

### Partly present, not claimed

These behaviours exist in the code, but they do not cover the catalogue description, so they are not counted as features. Where they matter to the teacher, they appear as acceptance criteria of the stories above.

| Code | Catalogue feature | What exists | What is missing | Covered in |
| --- | --- | --- | --- | --- |
| A3 | Human-in-the-loop | y/N confirmation before every network call (`web_search`, MCP servers marked `"network": true`), before every model-initiated `write_file`, before `/rag reset` and `/forget all`; Allow/Cancel in the web page with a 2-minute timeout. | No plan is presented before a multi-step action. | T3-02, T4-01, M1-03 |
| A4 | Prompt versioning | `monitoring.prompt_version` label stored with every metric and usable as a `/metrics --prompt-version` filter; `--prompt-version` in the benchmark script. | Prompt files are versioned by Git only: no timestamped versions, comparison, or rollback command. | EV3-01 |
| EV2 | Content and privacy guardrails | `personal_data_warning` flags e-mail addresses, phone numbers, IBANs, long numbers, pupil identifiers (INE), birth dates and support-plan words in text about to leave the machine; documents and tool output containing instructions addressed to the AI are flagged (`suspicious_instructions`); external output is framed as data, stripped of escape sequences and truncated (`tools/safety.py`). | No blocking or anonymisation of model input or output. | T3-02, R1-03 |
| EV4 | Adversarial testing | `scripts/probe_limits.py` (hallucination, codename, long prompt, injection in an indexed document) with PASS/FAIL verdicts; unit tests for a poisoned web snippet, forged markers, a hallucinated tool name, path traversal (files, ZIP entries, session identifiers); results in [limits](limits.md) and [security](security.md). | No large jailbreak suite. | T3-02, T4-01 |

## User stories

### B1 — Base loop

**B1-01 — Read the answer as it is written**

As a sixième mathematics teacher (Camille), I want each answer to appear progressively and to take my previous messages into account, so that I can follow up with "make it shorter" without repeating my request.

Acceptance criteria:
- After the question, `Assistant:` is printed and the text appears in fragments as Ollama streams it, not as one block at the end.
- A follow-up that refers to the previous answer ("Peux-tu raccourcir l'explication précédente ?") is answered in context: the request sent to the model contains the earlier user and assistant messages.
- A malformed stream line is reported as "Ollama could not complete this request. Please try again." and the prompt comes back.

Code: `cli/cli.py` (`stream_response`), `utilities/ollama_client.py` (`stream_message`) · Tests: `test_cli.py::TestCLIExceptionHandling::test_cli_successful_stream`, `test_ollama_client.py::TestOllamaHTTPClient::test_stream_message_success`, `test_cli.py::TestCLICommandDispatch::test_slash_load_restores_conversation_context_for_next_turn`, `test_cli.py::TestCLIExceptionHandling::test_cli_malformed_stream_response_displays_warning_without_crashing` · Shown: Demo 0:45 · Ask for the test date and room

**B1-02 — Keep working when something fails**

As a sixième mathematics teacher (Camille), I want SensAI to stay open when Ollama is stopped, the model is missing, or I press Enter on an empty line, so that a technical problem does not make me lose my preparation.

Acceptance criteria:
- With Ollama stopped, the message is "Connection error: unable to reach Ollama. Make sure Ollama is running, then try again." and no traceback is printed.
- If `qwen2.5:3b` is not installed, the line "Model 'qwen2.5:3b' is unavailable. Retrying with fallback 'llama3.2:latest'." is printed and the answer comes from the fallback model.
- An empty or blank line sends nothing to the model.
- After an error, the failed question is removed from the history and the next question works normally.

Code: `cli/error_handling.py`, `cli/fallback.py`, `cli/cli.py` · Tests: `test_cli.py::TestCLIExceptionHandling::test_cli_ollama_connection_error_does_not_exit`, `test_cli.py::TestCLIExceptionHandling::test_cli_mid_stream_disconnection_does_not_exit`, `test_model_fallback.py::TestFallbackChatClient::test_stream_retries_the_next_model_when_primary_is_missing`, `test_cli.py::TestCLICommandDispatch::test_empty_input_is_ignored`, `test_cli.py::TestCLIToolCalling::test_failed_turn_rolls_back_the_whole_turn` · Shown: doc only (fallback plan of the demo script)

**B1-03 — Choose the model per computer**

As a school IT technician who installs SensAI on teachers' laptops, I want to choose the chat model and its settings in `config.json`, so that each laptop runs a model that fits its memory without anyone editing code.

Acceptance criteria:
- Changing `model.chat_model` in `config.json` changes the model used at the next start; `temperature` and `context_window` are sent to Ollama as `temperature` and `num_ctx`.
- An invalid value (for example `"temperature": 3`) stops startup with a `Configuration error: ...` message before the chat opens.
- `fallback_chat_models` lists the models tried, in order, when the primary model is not installed.

Code: `config.py`, `cli/model_factory.py`, `main.py` · Tests: `test_config.py::TestConfigurationLoading::test_loads_a_complete_valid_configuration`, `test_config.py::TestConfigurationLoading::test_rejects_invalid_values_before_startup`, `test_config.py::TestApplicationStartup::test_main_reports_invalid_configuration_without_starting_the_cli`, `test_ollama_client.py::TestOllamaHTTPClient::test_chat_payload_includes_configured_ollama_options` · Shown: doc only ([configuration reference](configuration.md))

### R1 — Basic RAG

**R1-01 — Prepare from my own course**

As a sixième mathematics teacher (Camille), I want to index my fractions course and get explanations built from it, so that what I give my class uses the vocabulary and examples I already teach.

Acceptance criteria:
- `/ingest demo/fractions_6e.md` prints `Document indexed: fractions_6e.md (1 chunks; 1 new chunks stored).`
- `/rag on` prints `RAG mode enabled. Responses will use indexed documents only.`
- After a grounded answer, a line `Sources: fractions_6e.md (chunks: <chunk id>)` is printed by SensAI from the chunks it retrieved, not written by the model.
- After a restart, the course is still indexed (`data/sensai_rag.sqlite3`): no new `/ingest` is needed.

Code: `rag/context.py` (`RAGIndexer`, `RAGContextProvider`, `format_source_attribution`), `rag/storage.py`, `rag/embeddings.py` · Tests: `test_rag_integration.py::TestRAGIntegration::test_indexed_excerpt_is_injected_and_attributed_without_ollama`, `test_cli.py::TestCLIRAGIntegration::test_cli_displays_sources_used_for_a_rag_answer`, `test_storage.py::TestLocalVectorStore::test_chunks_remain_available_after_reopening_database`, `test_context.py::TestRAGIndexer::test_indexes_document_through_every_rag_stage` · Shown: Demo 0:25 · Index the course and enable RAG; Demo 0:45 · Ask for the test date and room

**R1-02 — Use the files I already have**

As a sixième mathematics teacher (Camille), I want to index the PDF and Word documents and the zipped chapter folders I already have, so that I do not retype my worksheets before using them.

Acceptance criteria:
- A text-based PDF and a `.docx` file are indexed; DOCX headings and paragraphs keep their order.
- `/ingest chapter.zip` prints `Archive indexed: chapter.zip (N files indexed; ...)`, lists skipped unsupported entries, and refuses entries with `..` or absolute paths.
- A scanned (image-only) PDF is refused with a message saying OCR is not supported; an unsupported file type is refused with a readable message.
- `/ingest --multiple a.pdf b.docx` indexes both and continues after one failure.

Code: `rag/ingestion.py`, `cli/commands/rag.py` · Tests: `test_ingestion.py::TestDocumentIngestor::test_ingests_text_based_pdf_with_document_metadata`, `test_ingestion.py::TestDocumentIngestor::test_ingests_docx_paragraphs_and_headings_in_order`, `test_ingestion.py::TestDocumentIngestor::test_ingests_supported_archive_entries_with_archive_metadata`, `test_ingestion.py::TestDocumentIngestor::test_archive_rejects_path_traversal_before_processing_entries`, `test_ingestion.py::TestDocumentIngestor::test_image_only_pdf_reports_that_ocr_is_not_supported`, `test_cli.py::TestCLIExceptionHandling::test_cli_ingest_multiple_indexes_quoted_paths_and_continues_after_a_failure` · Shown: doc only

**R1-03 — Be warned when a shared document talks to the AI**

As a sixième mathematics teacher (Camille), I want to be warned when a document shared by a colleague contains instructions addressed to the assistant, so that a hidden "SYSTEM:" line cannot quietly change my lesson preparation or make SensAI act without me.

Acceptance criteria:
- `/ingest demo/progression_piegee.md` prints `Warning: progression_piegee.md contains text that looks like instructions to an AI assistant (possible prompt injection)` and quotes the suspicious lines; the document is still indexed as data.
- When an excerpt of that document is used for an answer, `Warning: an excerpt used for this answer (progression_piegee.md) contains text addressed to an AI assistant` is printed.
- `/debug prompt` shows the excerpt inside `<<<DOCUMENT EXCERPTS>>>` in the user message, with the forged `[End of retrieved context]` neutralised and `SYSTEM:` quoted; no document text is in a system message.
- Any file write or network call the document asks for still requires a `[y/N]` answer.

Code: `tools/safety.py` (`suspicious_instructions`, `neutralize_markers`), `cli/commands/rag.py` (`_warn_about_instructions`), `cli/cli.py` (`injection_warning`, `_request_messages`), `rag/context.py` (`PreparedContext.user_message`) · Tests: `test_cli.py::TestIngestWarnsAboutInjectedInstructions::test_ingest_lists_lines_that_look_like_instructions`, `test_cli.py::TestCLIRequestAssembly::test_the_teacher_is_warned_when_an_excerpt_talks_to_the_ai`, `test_context.py::TestRAGContextProvider::test_a_document_cannot_close_the_frame_or_pose_as_a_role`, `test_tool_safety.py::TestSuspiciousInstructions::test_instructions_addressed_to_the_assistant_are_found` · Shown: Demo 1:45–2:00 · Poisoned document

### R2 — Advanced RAG (partial)

**R2-01 — Restrict retrieval to one chapter**

As a sixième mathematics teacher (Camille), I want to restrict retrieval to the fractions file, so that an explanation for my sixième class is not mixed with excerpts from another chapter or another level.

Acceptance criteria:
- `/rag filter name fractions_6e.md` prints `RAG filter set to source name: fractions_6e.md`, and the next `Sources:` line names only that file.
- `/rag filter category document` keeps only that category; an unknown category prints `Unknown category: ...` and keeps the previous filter.
- When no indexed chunk matches the filter, `No indexed chunks match the active RAG filter.` is printed.
- `/rag filter clear` prints `RAG metadata filter cleared.`

Code: `rag/storage.py` (`MetadataFilter`, SQL `WHERE` before scoring), `cli/commands/rag.py` · Tests: `test_cli.py::TestCLIExceptionHandling::test_rag_filter_commands_set_and_clear_metadata_restrictions`, `test_storage.py::TestLocalVectorStore::test_search_filters_chunks_by_source_filename_or_category`, `test_context.py::TestRAGContextProvider::test_metadata_filter_is_forwarded_to_vector_search`, `test_functional_prompts.py::TestPromptWorkflows::test_rag_prompt_applies_filter_and_can_hide_sources` · Shown: doc only

**R2-02 — Drop near-misses before they reach the model**

As a sixième mathematics teacher (Camille), I want weakly related excerpts to be dropped before the model sees them, so that an explanation of equivalent fractions is not polluted by a paragraph that only shares a few words with my question.

Acceptance criteria:
- Retrieved chunks whose cosine score is below `rag.relevance_threshold` (0.6 in `config.json`, calibrated on measured scores) are not sent; `/debug prompt` shows only the kept excerpts, each with its `Score:`.
- `logs/sensai.log` records the threshold and the selected and rejected chunk identifiers with their scores, without the chunk text.
- When every chunk is rejected and no tool fits the question, SensAI answers `Je ne trouve pas cette information dans les documents indexés.` without calling the model, and no `Sources:` line is printed.
- A threshold outside [-1, 1] is refused at startup.

Code: `rag/context.py` (`_filter_by_relevance`, `_log_retrieval`), `cli/cli.py` (`no_evidence_answer`) · Tests: `test_cli.py::TestCLIRequestAssembly::test_rag_without_relevant_excerpt_answers_without_calling_the_model`, `test_context.py::TestRAGContextProvider::test_relevance_threshold_keeps_only_sufficiently_similar_chunks`, `test_context.py::TestRAGContextProvider::test_relevance_threshold_builds_no_document_context_when_every_chunk_is_rejected`, `test_context.py::TestRAGContextProvider::test_relevance_threshold_configuration_is_validated`, `test_functional_prompts.py::TestPromptWorkflows::test_rag_prompt_without_relevant_context_does_not_claim_a_source` · Shown: Demo 1:30 · The "not found" guard

### M1 — Session persistence and profiles

**M1-01 — Resume tomorrow where I stopped**

As a sixième mathematics teacher (Camille), I want to close SensAI on Sunday evening and resume the same preparation on Monday, so that I can ask to "turn the previous explanation into a check question" without explaining everything again.

Acceptance criteria:
- `/save` prints `Session saved at sessions/<session id>.json`; `exit`, `quit`, Ctrl+C, and Ctrl+D also save when `features.memory_enabled` is true.
- `python3 source/main.py --session <session id>` restores the earlier user and assistant messages, and the next answer uses them.
- `/load <session id>` prints `Session <session id> loaded.`; an unknown identifier prints a `File error: ...` message and the chat continues.
- An identifier containing anything other than letters, digits, `-`, and `_` (for example `../x`) is refused with a `Session error: ...` message.

Code: `memory/session_manager.py`, `cli/commands/memory.py`, `cli/commands/system.py` · Tests: `test_cli.py::TestCLICommandDispatch::test_exit_command_saves_and_exits`, `test_cli.py::TestCLISessionLoading::test_valid_session_id_loads_messages`, `test_cli.py::TestCLICommandDispatch::test_slash_load_restores_conversation_context_for_next_turn`, `test_cli.py::TestCLICommandDispatch::test_slash_load_with_missing_session_prints_error`, `test_privacy.py::SessionIdTests::test_load_command_reports_an_unsafe_id_without_crashing` · Shown: Demo 2:55 · Save and exit; Demo 2:55 · Restart with the identifier; Demo 3:15 · Refer to the earlier answer

**M1-02 — Be known without repeating myself**

As a sixième mathematics teacher (Camille), I want SensAI to know once and for all that I teach mathematics to 11-year-olds and work in French, so that every answer is in French and pitched at my pupils without me saying it each time.

Acceptance criteria:
- Starting with `--profile demo/profile_camille.json` adds her name, preferred language, expertise, and constraints to the system message; `/debug prompt --include-system` shows them.
- Without a profile file, SensAI creates one with default values instead of failing.
- In the web page, saving the profile form changes the system message used by the next answer.

Code: `memory/profile_manager.py`, `cli/cli.py` (`_ensure_base_system_message`), `main.py` (`--profile`) · Tests: `test_cli.py::TestCLIPromptInjection::test_cli_injects_profile_system_prompt_on_new_session`, `test_profile_manager.py::TestProfileManager::test_build_system_prompt_structure`, `test_profile_manager.py::TestProfileManager::test_default_profile_creation_when_file_missing`, `test_web.py::ManagementTests::test_profile_update_changes_the_system_prompt` · Shown: Demo 0:00 · Scene, started with `--profile demo/profile_camille.json`

**M1-03 — Erase saved conversations**

As a school data protection officer, I want teachers to be able to erase saved conversations from inside SensAI, so that pupil information mentioned during a preparation is not kept indefinitely on their laptops.

Acceptance criteria:
- `/forget` deletes the saved file of the current session and clears the conversation from memory.
- `/forget all` asks `Delete ALL saved sessions? This cannot be undone. [y/N]` and deletes only on `y`, `yes`, `o`, or `oui`; it then prints how many sessions were deleted.
- Session and profile files are created readable by their owner only (mode `0600`, folder `0700`).

Code: `memory/session_manager.py` (`delete`, `delete_all`), `memory/private_files.py`, `cli/commands/memory.py` · Tests: `test_privacy.py::ForgetCommandTests::test_forget_deletes_this_session_and_clears_memory`, `test_privacy.py::ForgetCommandTests::test_forget_all_needs_confirmation`, `test_privacy.py::ForgetCommandTests::test_forget_all_deletes_every_session`, `test_privacy.py::PrivateFileTests::test_session_files_and_folder_are_owner_only` · Shown: doc only ([local data and GDPR assessment](RGPD.md))

### M2 — Token budgeting and semantic compression

**M2-01 — See how full the conversation is**

As a sixième mathematics teacher (Camille), I want to see how much of the model's context my preparation already uses, so that I know when a long conversation risks losing its beginning.

Acceptance criteria:
- `/tokens` prints the last turn (`prompt + completion`), the context use as `<tokens> / 4096 tokens (<percent>%)`, and the session total.
- From 80 % of `model.context_window`, a line `Warning: Context token usage is at <percent>% (<tokens>/4096 tokens).` follows the answer.
- A message that would not fit in the window minus the answer reserve is not sent: `This request needs about <N> tokens, but the context window is 4096 tokens ...` explains why and what to do.
- With `features.memory_enabled` set to false, `/tokens` prints `Memory is disabled in config.json.`

Code: `cli/session.py` (`record_turn_tokens`, `get_budget_warning`), `cli/commands/memory.py` · Tests: `test_cli.py::TestCLICommandDispatch::test_tokens_command_tracks_per_turn_and_cumulative_usage`, `test_cli.py::TestCLICommandDispatch::test_context_budget_warning_displayed_when_threshold_exceeded`, `test_cli.py::TestCLICommandDispatch::test_context_budget_warning_not_displayed_when_under_threshold`, `test_cli.py::TestCLICommandDispatch::test_cli_session_budget_warning_thresholds`, `test_cli.py::TestCLIRequestAssembly::test_a_request_larger_than_the_window_is_refused_before_sending`, `test_context_budget.py::TestContextBudget::test_refusal_message_explains_the_risk_and_the_remedies` · Shown: Demo 3:35 · `/tokens` and `/metrics`

**M2-02 — Keep a long preparation in one conversation**

As a sixième mathematics teacher (Camille), I want older exchanges to be condensed automatically during a long preparation, so that I can keep working in the same conversation while the decisions taken at the start (level, chapter, vocabulary) are kept.

Acceptance criteria:
- At 85 % context use, `Auto-compression triggered: condensed older turns (<before> -> <after> messages).` is printed after the answer; when a new request would not fit, `Context nearly full: older turns were summarised before sending` is printed before it is sent.
- The summary keeps the teacher's name, class, numbers, dates, decisions and open tasks.
- `/compress` runs the same compression on demand and prints the number of messages before and after.
- The persona and profile system message and the two most recent exchanges are kept word for word; older exchanges become one summary message.
- If the summary call fails or returns nothing, the history is left unchanged and the reason is printed.

Code: `memory/compression.py`, `cli/cli.py`, `cli/commands/memory.py` · Tests: `test_compression.py::TestSemanticCompression::test_compress_history_preserves_base_system_and_recent_turns`, `test_compression.py::TestSemanticCompression::test_compress_history_handles_ollama_error_gracefully`, `test_compression.py::TestSemanticCompression::test_compress_history_with_existing_summary`, `test_cli.py::TestCLICommandDispatch::test_auto_compression_triggered_when_budget_critical`, `test_cli.py::TestCLIRequestAssembly::test_history_is_summarised_before_sending_when_it_would_overflow` · Shown: doc only (web button "Summarise older messages")

### M4 — Artifact and state document

**M4-01 — Build the lesson plan as a separate document**

As a sixième mathematics teacher (Camille), I want SensAI to write my lesson plan as a separate document, section by section, so that I end up with a clean plan (objectives, starter, activity, homework) instead of copying fragments out of a long chat.

Acceptance criteria:
- A request containing "plan" (for example "Prépare un plan de séance sur les fractions équivalentes") lets the model call `edit_artifact`; each call is printed as `[tool] edit_artifact({...})`.
- `/artifact view` prints the document as Markdown (`#` title, `##` sections); `/artifact outline` lists the section titles.
- Asking to change only the homework section updates that section (`replace`, `append`, or `prepend`) and leaves the others unchanged.

Code: `memory/artifact_manager.py`, `tools/artifact_tool.py`, `cli/commands/artifact.py` · Tests: `test_artifact_tool.py::TestArtifactTool::test_execute_update_section`, `test_artifact_manager.py::TestArtifactManagerEditing::test_update_section_append_mode`, `test_artifact_manager.py::TestArtifactManagerEditing::test_replace_text_in_section`, `test_artifact_tool.py::TestArtifactTool::test_is_relevant_for`, `test_cli.py::TestCLIArtifactCommands::test_artifact_init_and_outline` · Shown: doc only

**M4-02 — Find the plan again with the conversation**

As a sixième mathematics teacher (Camille), I want the lesson plan to be saved and reopened with its conversation, so that on Monday I continue the same plan rather than starting a new one.

Acceptance criteria:
- On `exit` (or `/save`), the plan is written next to the session as `sessions/<session id>_artifact.md` and `.json`.
- `--session <session id>` or `/load <session id>` reloads the plan; `/load` prints `Artifact for session <session id> loaded.`
- In the web page, the plan's title, version, and sections are listed, and a new conversation starts without a plan.

Code: `memory/artifact_manager.py` (`save`, `load`), `cli/cli.py`, `cli/commands/memory.py`, `web/app.py` · Tests: `test_cli.py::TestArtifactSessionLifecycle::test_save_and_load_session_with_artifact`, `test_cli.py::TestArtifactSessionLifecycle::test_exit_command_auto_saves_artifact`, `test_artifact_manager.py::TestArtifactManagerPersistence::test_save_and_load_artifact`, `test_web.py::ArtifactAndModelTests::test_living_document_is_exposed_and_saved_with_the_conversation`, `test_web.py::ArtifactAndModelTests::test_new_conversation_discards_the_document` · Shown: doc only

### T1 — MCP

**T1-01 — Exact class calculations**

As a sixième mathematics teacher (Camille), I want SensAI to compute a pupil's weighted average from the marks and coefficients I give, so that I get an exact result instead of mental arithmetic from a language model.

Acceptance criteria:
- `/tools` lists `teacher__calculate_grade`, `teacher__class_statistics`, `teacher__start_timer`, and `teacher__timer_status`, each marked `[teacher]` as its origin.
- "Calcule la moyenne pondérée de 12, 15 et 9 avec les coefficients 1, 2 et 1" produces a `[tool] teacher__calculate_grade({...})` line, then an answer using the tool result.
- Invalid input (a mark above the maximum, a missing list) comes back to the model as an error result; SensAI keeps running.

Code: `mcp/servers/teacher_tools.py`, `mcp/client.py`, `mcp/manager.py`, `cli/tool_calls.py`, `tools/registry.py` · Tests: `test_teacher_tools_server.py::TestCalculateGrade::test_weighted_average`, `test_teacher_tools_server.py::TestTeacherToolsWithSensai::test_tool_call_through_the_registry`, `test_teacher_tools_server.py::TestCalculateGrade::test_invalid_arguments_are_reported_with_is_error`, `test_cli.py::TestCLIMCPTools::test_tools_command_lists_native_and_mcp_tools` · Shown: Demo 2:20 · MCP grade tool (`teacher__calculate_grade`)

**T1-02 — Plug in an existing tool server**

As a school IT technician, I want to add an existing MCP server by editing `mcp_servers.json` only, so that teachers gain new tools without a change to SensAI's code, and a broken server never blocks the chat.

Acceptance criteria:
- A server declared in `mcp_servers.json` has its tools listed by `/tools` at the next start, prefixed `<server>__` (the repository declares `teacher` and the third-party `web` server, `uvx mcp-server-fetch`).
- A server that fails to start prints a `Warning: ...` line at startup; the other servers' tools stay available.
- A server marked `"network": true` starts only when `features.network_enabled` is true.
- A server process receives only an allow-list of environment variables (`PATH`, `HOME`, `LANG`, ...), not the user's tokens or keys.

Code: `mcp/config.py`, `mcp/manager.py`, `mcp/client.py` (`safe_environment`) · Tests: `test_mcp_manager.py::TestMCPManager::test_broken_server_does_not_prevent_the_others`, `test_mcp_third_party.py::TestThirdPartySchemaCompatibility::test_call_reaches_the_server_with_the_original_tool_name`, `test_web_search.py::NetworkSwitchTests::test_network_servers_are_skipped_by_default`, `test_privacy.py::McpEnvironmentTests::test_secrets_are_dropped_and_runtime_variables_kept`; opt-in against the real server: `test_mcp_third_party.py::TestRealThirdPartyServer::test_both_servers_run_together` (`SENSAI_THIRD_PARTY_TESTS=1`, requires `uvx`) · Shown: doc only

### T3 — Web search

**T3-01 — Check a recent source with its links**

As a sixième mathematics teacher (Camille), I want to ask SensAI to look something up on the web and see the links it used, so that I can check the source myself before using it in class.

Acceptance criteria:
- With `features.network_enabled: true`, startup prints `Network access is enabled: web search and web servers can send requests to the internet.`
- A request that explicitly mentions the web ("cherche sur le web ...") offers the `web_search` tool and skips document retrieval for that turn.
- After the answer, `Web sources consulted:` lists the result URLs given to the model.
- A network failure or a blocked request does not stop SensAI: the model is told the web could not be searched and answers without it.

Code: `tools/web_search.py`, `cli/cli.py` (`_take_web_sources`) · Tests: `test_web_search.py::WebSearchToolTests::test_returns_formatted_results`, `test_web_search.py::RoutingTests::test_explicit_web_requests_are_offered_and_claimed`, `test_web_search.py::NetworkSwitchTests::test_take_web_sources_reads_and_resets`, `test_web_search.py::WebSearchToolTests::test_network_failure_becomes_a_tool_error_with_fallback_instruction` · Shown: doc only

**T3-02 — Approve what leaves my laptop**

As a sixième mathematics teacher (Camille) who sometimes mentions pupils in her questions, I want to see and approve the exact text before anything is sent to a search engine, so that no pupil information leaves my laptop without my explicit yes.

Acceptance criteria:
- With the default `config.json` (`network_enabled: false`), the `web_search` tool is not registered at all.
- Before each search, SensAI asks `Search the web for "<query>"? This text is sent to DuckDuckGo. [y/N]`; any answer other than `y`, `yes`, `o`, or `oui` sends nothing.
- If the query contains an e-mail address, a phone number, an IBAN, or a long number, the question ends with `Warning: this looks like it contains ...`.
- In the web page, the same question appears with Allow and Cancel; no answer within two minutes counts as Cancel.

Code: `tools/web_search.py` (`confirmation_prompt`), `tools/safety.py` (`personal_data_warning`), `cli/tool_calls.py`, `web/app.py` (`_confirm`) · Tests: `test_web_search.py::NetworkSwitchTests::test_web_search_is_not_registered_by_default`, `test_web_search.py::ConfirmationTests::test_question_shows_the_exact_query`, `test_web_search.py::ConfirmationTests::test_declined_search_sends_nothing`, `test_web_search.py::ConfirmationTests::test_only_yes_answers_confirm`, `test_privacy.py::PersonalDataWarningTests::test_confirmation_question_carries_the_warning`, `test_web.py::ToolConfirmationTests::test_refused_tool_does_not_run` · Shown: doc only

### T4 — File access within permissions

**T4-01 — Read my courses, write only to exports**

As a sixième mathematics teacher (Camille), I want SensAI to read files in my `courses/` folder and write only into `exports/`, so that it can use and save my material without being able to touch anything else on my laptop.

Acceptance criteria:
- `/permissions` prints `File access rules:` followed by `courses` (read) and `exports` (read, write).
- Asking to read a file in `courses/` produces a `[tool] read_file({...})` line and an answer using its content.
- Writing into `courses/`, reading outside the allowed folders, a `../` path, or a symbolic link leaving an allowed folder are refused with `Access denied: ...`.
- `.env`, `.git`, `*.pem`, `*.key`, and `id_rsa*` are refused even inside an allowed folder; an existing file is never overwritten unless the model sets `overwrite: true`.
- Every `write_file` asks `Write <n> characters to '<path>'? ... [y/N]` first, with `It REPLACES the existing file.` when it overwrites; answering no (or nothing) leaves the file unchanged.

Code: `tools/permissions.py`, `tools/filesystem.py`, `permissions.json` · Tests: `test_permissions.py::TestPermissionChecker::test_read_only_directory_refuses_writes`, `test_permissions.py::TestPermissionChecker::test_path_traversal_is_blocked`, `test_permissions.py::TestPermissionChecker::test_deny_rules_win_over_allow`, `test_filesystem_tools.py::TestPermissionCheckComesFirst::test_no_operation_runs_when_the_permission_check_fails`, `test_filesystem_tools.py::TestWriteFile::test_existing_file_is_kept_unless_overwrite`, `test_permissions.py::TestPermissionsCommand::test_permissions_command_lists_the_rules`, `test_filesystem_tools.py::TestWriteConfirmation::test_a_refused_write_changes_nothing` · Shown: doc only

**T4-02 — Deny by default**

As a school IT technician, I want file access to be denied unless one JSON file allows it, so that a missing or broken `permissions.json` never opens the whole disk to the model.

Acceptance criteria:
- Without `permissions.json`, no file tool can read or write, and `/permissions` prints `No file access is allowed.`
- An unreadable or invalid file prints a startup warning ending with `All file access is denied.`
- An invalid entry is skipped with a warning; the valid entries still apply.
- Relative folders are resolved from the location of `permissions.json`, not from the current directory.

Code: `tools/permissions.py` (`load_permissions`, `PermissionChecker`) · Tests: `test_permissions.py::TestLoadPermissions::test_missing_file_denies_everything`, `test_permissions.py::TestLoadPermissions::test_unreadable_file_denies_everything_and_warns`, `test_permissions.py::TestLoadPermissions::test_invalid_entries_are_skipped_with_a_warning`, `test_permissions.py::TestPermissionChecker::test_relative_paths_are_relative_to_the_project_not_the_current_directory`, `test_permissions.py::TestPermissionsCommand::test_permission_warnings_are_displayed_at_startup` · Shown: doc only

### T5 — Structured output

**T5-01 — A quiz from my course, always in the same shape**

As a sixième mathematics teacher (Camille), I want to generate a multiple-choice quiz with exactly the number of questions I ask for, from my indexed course, so that I get a ready-to-check quiz in the same format every time.

Acceptance criteria:
- `/quiz --questions 5 Les fractions équivalentes` prints a title, five numbered questions with lettered choices (`A)` to `F)`), then `Answer key:` with one letter and explanation per question.
- A model answer with the wrong number of questions is rejected and requested again, up to three attempts.
- With RAG on and no matching excerpt, SensAI prints `No indexed course excerpt matches this topic. Index a document with /ingest or turn RAG off.` and generates nothing.
- After three invalid answers, `Quiz generation failed: The model did not produce a valid answer after 3 attempts: ...` is printed and the chat continues.

Code: `tools/structured_output.py` (`generate_structured`, `check_quiz`, `quiz_messages`), `cli/commands/tools.py` · Tests: `test_structured_output.py::TestStructuredOutputCommands::test_quiz_is_built_from_the_course_when_rag_is_on`, `test_structured_output.py::TestStructuredOutputCommands::test_quiz_with_rag_on_but_no_matching_excerpt`, `test_quiz_export.py::TestQuizCommand::test_wrong_number_of_questions_is_retried`, `test_structured_output.py::TestGenerateStructured::test_gives_up_after_the_maximum_number_of_attempts` · Shown: doc only

**T5-02 — An answer key that points to a real choice**

As a sixième mathematics teacher (Camille), I want every quiz to be checked against a fixed schema before I see it, so that I never distribute a question whose answer key points to a choice that does not exist.

Acceptance criteria:
- The JSON schema is sent to Ollama as the `format` of the request, and the answer is validated again locally (types, 2 to 6 choices, required fields).
- An `answer_index` outside the list of choices, invalid JSON, or a schema violation triggers a new attempt that includes the error message.
- The quiz is written in the language given by `--language`, otherwise in the profile's preferred language.
- Limit stated to the teacher: the check guarantees the structure, not that the marked answer is mathematically correct.

Code: `tools/structured_output.py` (`QUIZ_SCHEMA`, `validate_json`), `utilities/ollama_client.py` (`output_format`) · Tests: `test_structured_output.py::TestGenerateStructured::test_valid_answer_is_returned_and_schema_is_sent_to_ollama`, `test_structured_output.py::TestGenerateStructured::test_schema_violation_is_retried_with_the_error`, `test_structured_output.py::TestGenerateStructured::test_invalid_json_is_retried_with_the_error`, `test_structured_output.py::TestQuizHelpers::test_check_quiz`, `test_quiz_export.py::TestQuizCommand::test_quiz_language_follows_the_option_then_the_profile` · Shown: doc only

### X3 — Export and publication (partial)

**X3-01 — A pupil copy and a teacher copy**

As a sixième mathematics teacher (Camille), I want to save the quiz once without the answer key and once with it, so that I print the first for the class and keep the second for marking.

Acceptance criteria:
- `/quiz --save exports/fractions-eleves.txt --no-answers Les fractions` prints `Quiz saved to 'exports/fractions-eleves.txt' (without the answer key).`; the file has no `Answer key:` section.
- A `.json` path saves JSON; with `--no-answers`, only `answer_index` and `explanation` are removed.
- An existing file is kept unless `--overwrite` is given; a path outside the permitted folders is refused, and the quiz stays displayed.
- In the web page, the buttons "With answers", "For students", and "JSON" download the last quiz.

Code: `tools/quiz_export.py` (`render_quiz`, `save_quiz`), `cli/commands/tools.py`, `web/server.py` · Tests: `test_quiz_export.py::TestQuizCommand::test_student_copy`, `test_quiz_export.py::TestRenderQuiz::test_json_without_answers_drops_the_answer_fields_only`, `test_quiz_export.py::TestQuizCommand::test_existing_file_needs_overwrite`, `test_quiz_export.py::TestQuizCommand::test_refused_path_reports_the_error_and_keeps_the_quiz_displayed`, `test_web.py::ManagementTests::test_last_quiz_can_be_rendered_without_answers` · Shown: doc only

**X3-02 — Take the lesson plan out of SensAI**

As a sixième mathematics teacher (Camille), I want to export my lesson plan as a Markdown file, so that I can open it in my word processor or send it to a colleague.

Acceptance criteria:
- `/artifact export exports/plan-fractions.md` writes the document and prints `Artifact exported successfully to <path>`.
- An existing file is kept unless `--overwrite` is given (`Cannot export: File '...' already exists. Use --overwrite to replace it.`).
- A path outside the folders allowed by `permissions.json` is refused with `Cannot export: Access denied: ...`.
- In the web page, the living document can be downloaded as Markdown.

Code: `cli/commands/artifact.py`, `memory/artifact_manager.py` (`render_markdown`), `web/server.py` (`/api/artifact/download`) · Tests: `test_cli.py::TestCLIArtifactCommands::test_artifact_export`, `test_cli.py::TestCLIArtifactCommands::test_artifact_export_without_overwrite_refuses`, `test_cli.py::TestCLIArtifactCommands::test_artifact_export_permission_denied`, `test_web.py::ServerTests::test_artifact_download` · Shown: doc only

### A5 — Persona

**A5-01 — Switch to the mathematics assistant mid-conversation**

As a sixième mathematics teacher (Camille), I want to switch from the general teacher assistant to the mathematics assistant during a conversation, so that mathematics-specific instructions apply to my next questions without losing what we already said.

Acceptance criteria:
- `/persona list` lists `teacher-assistant` and `mathematics-assistant` and marks the active one `(active)`; `/persona status` prints its role, tone, and scope.
- `/persona use mathematics-assistant` prints `Active persona changed to: Mathematics assistant (mathematics-assistant)`; the earlier user and assistant messages are still sent with the next question.
- An unknown identifier prints `Unknown persona: <id>. Available personas: ...` and changes nothing.

Code: `cli/commands/persona.py`, `config.py` (`PersonasConfig`) · Tests: `test_cli.py::TestPersonaCommands::test_persona_commands_list_status_and_switch_without_erasing_history`, `test_cli.py::TestPersonaCommands::test_next_model_request_uses_the_runtime_selected_persona` · Shown: Demo 0:00 · Scene and profile; Demo 0:15 · Select the mathematics persona

**A5-02 — A persona for another subject without code**

As a French teacher in Camille's school, I want a "French assistant" to be added by editing `config.json` and a prompt text file, so that the same tool serves my subject without any code change.

Acceptance criteria:
- A new entry in `personas.profiles` with `identifier`, `display_name`, `role`, `tone`, `response_scope`, and either `system_prompt` or `prompt_file` appears in `/persona list` after a restart.
- An entry with both prompt sources, or neither, stops startup with a configuration error.
- A `prompt_file` outside the configuration folder, missing, empty, or not UTF-8 is refused at startup.

Code: `config.py` (`_read_persona`, `_read_prompt_file`), `prompts/*.txt` · Tests: `test_config.py::TestConfigurationLoading::test_loads_a_persona_prompt_from_a_file_relative_to_the_configuration`, `test_config.py::TestConfigurationLoading::test_rejects_personas_with_both_or_no_prompt_source`, `test_config.py::TestConfigurationLoading::test_loads_and_selects_a_configured_persona` · Shown: doc only ([personas](features/personas.md))

### EV3 — Logging and monitoring dashboard

**EV3-01 — Diagnose a slow laptop without reading conversations**

As a school IT technician supporting teachers, I want local metrics on model calls without any conversation text, so that I can find out why SensAI is slow or failing on a laptop without seeing pupils' data.

Acceptance criteria:
- With `monitoring.enabled: true`, `/metrics` prints `Local monitoring dashboard`, then model calls, average and p95 latency, error rate, RAG retrievals, estimated tokens, tool calls (succeeded and failed), and calls per day.
- Filters `--from`, `--to`, `--model`, `--prompt-version`, `--feature rag|tools|memory`, `--tool`, and `--error` combine; an unknown option is refused.
- The metrics database has no column for prompts, responses, excerpts, or tool arguments.
- With monitoring disabled (`monitoring.enabled: false`), `/metrics` prints `Monitoring is disabled in config.json.` and no database file is created. The repository's `config.json` enables it.

Code: `observability/monitoring.py`, `cli/commands/metrics.py`, `cli/cli.py` (`_record_model_interaction`) · Tests: `test_monitoring.py::TestMonitoringStore::test_persists_aggregates_and_filters_safe_interaction_metadata`, `test_monitoring.py::TestMonitoringStore::test_database_schema_has_no_raw_prompt_or_response_columns`, `test_monitoring.py::TestMonitoringStore::test_disabled_monitoring_does_not_create_a_database`, `test_monitoring.py::TestMonitoringCliIntegration::test_enabled_monitoring_records_each_model_request_without_storing_text` · Shown: Demo 3:35 · `/tokens` and `/metrics`

**EV3-02 — Understand a surprising answer**

As a sixième mathematics teacher (Camille), I want to see exactly what was sent to the model when an answer surprises me, so that I can tell whether the problem comes from my document or from the model.

Acceptance criteria:
- `/debug prompt` prints `Reconstructed model request:`, the `Settings:` sent (model, temperature, `num_ctx`) and the `Tools offered`, then numbered, labelled messages: `RAG rules`, `User message with document excerpts (framed as data)`, `User message`, `Assistant history`, `Conversation summary`, `Tool message`.
- `/stats` prints the speed Ollama measured for the last answer (generation and prompt-reading tokens per second, load time).
- The system message (persona and profile) is replaced by `<hidden; use /debug prompt --include-system to display it>` unless `--include-system` is given.
- `logs/sensai.log` lines carry a timestamp, a level, and a component; nothing from the log is printed in the chat.

Code: `cli/prompt_inspection.py`, `cli/commands/debug.py`, `observability/app_logging.py` · Tests: `test_observability.py::TestPromptInspection::test_debug_prompt_displays_user_assistant_and_tool_messages`, `test_observability.py::TestPromptInspection::test_prompt_formatter_displays_system_content_only_when_explicitly_requested`, `test_observability.py::TestPromptInspection::test_prompt_formatter_identifies_rag_context`, `test_observability.py::TestApplicationLogging::test_logs_include_timestamp_level_and_component_without_console_output`, `test_cli.py::TestCLIRequestAssembly::test_debug_prompt_lists_settings_and_offered_tools`, `test_cli.py::TestCLIRequestAssembly::test_stats_reports_the_speed_measured_by_ollama` · Shown: Demo 1:05 · `/debug prompt`; Demo 2:40 · `/stats`

### X1 — Web UI

**X1-01 — Use SensAI without a terminal**

As a primary-school teacher in Camille's school who has never used a terminal, I want to use SensAI from a web page on my own computer, so that I can add my course, answer from it, and generate a quiz with buttons instead of commands.

Acceptance criteria:
- `python3 source/main.py --web` opens `http://localhost:8000`; answers appear progressively in the page.
- "Add a document" uploads and indexes a file; "Answer from my documents" turns RAG on or off; "Show which document was used" controls the sources line.
- The quiz form (topic, number of questions, language) generates a quiz.
- A stopped Ollama server or a missing model is explained on the page, and the page keeps working.

Code: `web/server.py`, `web/app.py`, `web/static/index.html` · Tests: `test_web.py::ChatTests::test_answer_is_streamed_and_kept_in_history`, `test_web.py::ServerTests::test_chat_streams_json_lines`, `test_web.py::ManagementTests::test_upload_is_indexed_then_deleted`, `test_web.py::ChatTests::test_rag_context_is_used_and_sources_are_reported`, `test_web.py::ArtifactAndModelTests::test_model_check_reports_a_missing_model_and_a_stopped_ollama` · Shown: doc only

**X1-02 — Manage a long preparation from the page**

As a sixième mathematics teacher (Camille), I want to see context usage and switch persona or conversation from the page, so that I manage a preparation the same way as in the terminal.

Acceptance criteria:
- The page shows a context meter in tokens out of `model.context_window` and a "Summarise older messages" button.
- A persona selector switches persona; slash commands typed in the page (for example `/rag on`) run the same handlers as in the terminal.
- "New conversation" saves the current one and starts empty; "Delete all saved conversations" asks for confirmation before deleting.

Code: `web/app.py` (`state`, `run_command`, `new_conversation`, `forget_all`) · Tests: `test_web.py::ManagementTests::test_state_lists_documents_tools_and_tokens`, `test_web.py::ChatTests::test_slash_command_runs_through_the_cli_handlers`, `test_web.py::ManagementTests::test_new_conversation_saves_the_old_one_and_starts_empty`, `test_web.py::ManagementTests::test_forget_all_needs_the_browser_confirmation` · Shown: doc only

**X1-03 — Reachable from this computer only**

As a school IT technician, I want the web page to answer only on the teacher's own computer, so that another device on the school network or another website open in the browser cannot read or drive her conversations.

Acceptance criteria:
- The server listens on `127.0.0.1` only.
- A request whose `Host` header is not `localhost:<port>` or `127.0.0.1:<port>` gets `403` and `Unknown host.`
- A `POST` that is not JSON (or raw bytes for uploads) is refused, so a form on another site cannot submit to it; an upload larger than 25 MB is refused.

Code: `web/server.py` (`HOST`, `_check_host`, `_read_body`) · Tests: `test_web.py::ServerTests::test_foreign_host_is_refused`, `test_web.py::ServerTests::test_post_must_be_json_or_raw_bytes`, `test_web.py::ServerTests::test_page_and_state_are_served` · Shown: doc only ([web interface](features/web-interface.md))

### Count per feature

| B1 | R1 | R2 | M1 | M2 | M4 | T1 | T3 | T4 | T5 | X3 | A5 | EV3 | X1 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 3 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 32 |

### Former story identifiers

The first version of [products-scenario.md](products-scenario.md) had 14 stories. They are rewritten above with the subject's exact format and one feature each.

| Former ID | Now | Change |
| --- | --- | --- |
| RAG-01 | R1-01 | Persona made specific. |
| RAG-02 | R2-01 | Metadata filtering is R2, not R1. |
| TOOLS-01 | T5-01 and X3-01 | Split: generating the quiz (T5) and saving copies (X3) were one story spanning T5, X3, T4, and R1. |
| TOOLS-02 | T1-01 and T4-01 | Split: MCP calculations and file permissions. |
| PERSONA-01 | A5-01 | Unchanged scope. |
| PERSONA-02 | A5-02 | "Maintainer" replaced by a teacher of another subject. |
| MEMORY-01 | M1-01 | Unchanged scope. |
| MEMORY-02 | M2-01 and M2-02 | Split: budget display and compression. |
| OBS-01 | EV3-02 | "Developer" replaced by the teacher. |
| OBS-02 | EV3-01 | "Developer" replaced by the school IT technician. |
| WEB-01 | X1-01 | Unchanged scope. |
| WEB-02 | T3-02 | It describes web-search approval (T3), not the web page. |
| CLI-01 | B1-03 | "Teacher or maintainer" replaced by the school IT technician. |
| CLI-02 | B1-02 | Unchanged scope. |

## Brief to implementation

### Requirements of the subject

| Requirement (`subjects/Sensai_project.pdf`) | Feature | Code | Test | Where it is shown |
| --- | --- | --- | --- | --- |
| Streaming responses | B1 | `cli/cli.py` (`stream_response`), `utilities/ollama_client.py` (`stream_message`) | `test_cli.py::TestCLIExceptionHandling::test_cli_successful_stream` | Every demo answer |
| In-session history sent to the model | B1 | `cli/cli.py` (`messages`) | `test_cli.py::TestCLICommandDispatch::test_slash_load_restores_conversation_context_for_next_turn` | Demo 3:15 |
| Error handling: unavailable model | B1 | `cli/fallback.py`, `cli/error_handling.py` | `test_model_fallback.py::TestFallbackChatClient::test_stream_retries_the_next_model_when_primary_is_missing`, `test_cli.py::TestCLIExceptionHandling::test_cli_generic_ollama_error_displays_warning` | Story B1-02 |
| Error handling: empty input | B1 | `cli/cli.py` | `test_cli.py::TestCLICommandDispatch::test_empty_input_is_ignored` | Story B1-02 |
| Error handling: connection failure | B1 | `utilities/ollama_client.py`, `cli/error_handling.py` | `test_cli.py::TestCLIExceptionHandling::test_cli_ollama_connection_error_does_not_exit`, `test_cli.py::TestCLIExceptionHandling::test_cli_mid_stream_disconnection_does_not_exit` | Story B1-02 |
| Model selectable without code change | B1 | `config.json` → `model.chat_model`, `config.py` | `test_config.py::TestConfigurationLoading::test_loads_a_complete_valid_configuration` | [Configuration reference](configuration.md) |
| No LLM framework, no Ollama SDK | All | `requirements.txt` (only `requests`, `pypdf`, `python-docx`, plus `coverage` and `ruff` for development); every model call in `utilities/ollama_client.py` | `test_ollama_client.py` (mocks `requests`) | [Decisions](decisions.md) |
| `README.md` with setup, features, examples | — | `README.md`, [user guide](user-guide.md), [feature documentation](features/README.md) | — | — |
| `requirements.txt` | — | `requirements.txt` | — | — |
| At least two user stories per feature | — | This document | — | [Count per feature](#count-per-feature) |
| Keynote: concrete scenario | — | [Product scenario](products-scenario.md), [demo script](../demo/demo-script.md) | — | Demo |
| Keynote: what was considered and ruled out | — | [Decisions](decisions.md) | — | Keynote |

### Needs of the scenario

The needs come from Camille's situation as described in [products-scenario.md](products-scenario.md).

| Camille's need | Feature | Code | Test | Story | Where it is shown |
| --- | --- | --- | --- | --- | --- |
| Answers built from her own course, with the source named | R1 | `rag/` | `test_rag_integration.py::TestRAGIntegration::test_indexed_excerpt_is_injected_and_attributed_without_ollama` | R1-01 | Demo 0:25–1:05 |
| Only the relevant chapter, no near-misses | R2 | `rag/storage.py`, `rag/context.py` | `test_context.py::TestRAGContextProvider::test_relevance_threshold_keeps_only_sufficiently_similar_chunks` | R2-01, R2-02 | Doc only |
| French, sixième level, without repeating it | M1, A5 | `memory/profile_manager.py`, `config.py` | `test_cli.py::TestCLIPromptInjection::test_cli_injects_profile_system_prompt_on_new_session` | M1-02, A5-01 | Demo 0:00–0:25 |
| Resume a preparation the next day | M1 | `memory/session_manager.py` | `test_cli.py::TestCLISessionLoading::test_valid_session_id_loads_messages` | M1-01 | Demo 2:55–3:15 |
| Long preparation in a 4,096-token window | M2 | `cli/session.py`, `memory/compression.py` | `test_compression.py::TestSemanticCompression::test_compress_history_preserves_base_system_and_recent_turns` | M2-01, M2-02 | Demo 3:35 |
| A lesson plan as a document, not a chat | M4 | `memory/artifact_manager.py`, `tools/artifact_tool.py` | `test_cli.py::TestArtifactSessionLifecycle::test_save_and_load_session_with_artifact` | M4-01, M4-02 | Doc only |
| A quiz in a fixed, checkable format | T5 | `tools/structured_output.py` | `test_structured_output.py::TestStructuredOutputCommands::test_quiz_is_built_from_the_course_when_rag_is_on` | T5-01, T5-02 | Doc only |
| A pupil copy without answers | X3, T4 | `tools/quiz_export.py`, `tools/permissions.py` | `test_quiz_export.py::TestQuizCommand::test_student_copy` | X3-01 | Doc only |
| Exact averages and class statistics | T1 | `mcp/servers/teacher_tools.py` | `test_teacher_tools_server.py::TestCalculateGrade::test_weighted_average` | T1-01 | Demo 2:20 · MCP grade tool |
| Pupil data never leaves the laptop without her consent | T3 (and local Ollama) | `tools/web_search.py`, `tools/safety.py`, `config.json` → `network_enabled` | `test_web_search.py::ConfirmationTests::test_declined_search_sends_nothing` | T3-02, M1-03 | Demo 4:00 (`/forget` and spoken) |
| The assistant only touches her course and export folders | T4 | `tools/permissions.py` | `test_permissions.py::TestPermissionChecker::test_path_traversal_is_blocked` | T4-01 | Doc only |
| Colleagues without a terminal | X1 | `web/` | `test_web.py::ChatTests::test_answer_is_streamed_and_kept_in_history` | X1-01 | Doc only |
| Runs on her laptop at an acceptable speed | Model choice | `config.json` → `model` | `scripts/benchmark_models.py` (measurement, not a unit test) | B1-03 | [Hardware feasibility](products-scenario.md#hardware-feasibility) |
| Find out why it is slow or wrong | EV3 | `observability/` | `test_monitoring.py::TestMonitoringCliIntegration::test_enabled_monitoring_records_each_model_request_without_storing_text` | EV3-01, EV3-02 | Demo 3:35 |

## Jury exercise

Give a juror one of these stories without its identifier and ask which feature it comes from. The answers are below the table.

| # | Story |
| --- | --- |
| 1 | As a sixième mathematics teacher (Camille) who sometimes mentions pupils in her questions, I want to see and approve the exact text before anything is sent to a search engine, so that no pupil information leaves my laptop without my explicit yes. |
| 2 | As a sixième mathematics teacher (Camille), I want to restrict retrieval to the fractions file, so that an explanation for my sixième class is not mixed with excerpts from another chapter or another level. |
| 3 | As a sixième mathematics teacher (Camille), I want older exchanges to be condensed automatically during a long preparation, so that I can keep working in the same conversation while the decisions taken at the start are kept. |
| 4 | As a sixième mathematics teacher (Camille), I want to save the quiz once without the answer key and once with it, so that I print the first for the class and keep the second for marking. |
| 5 | As a sixième mathematics teacher (Camille), I want SensAI to compute a pupil's weighted average from the marks and coefficients I give, so that I get an exact result instead of mental arithmetic from a language model. |

Answers:

1. **T3 Web search** (T3-02). Not X1: the approval exists in the terminal as well as in the page. Proof: `test_web_search.py::ConfirmationTests::test_declined_search_sends_nothing`.
2. **R2 Advanced RAG, metadata filtering** (R2-01). Not R1: basic retrieval has no filter. Proof: `/rag filter name fractions_6e.md`, `test_storage.py::TestLocalVectorStore::test_search_filters_chunks_by_source_filename_or_category`.
3. **M2 Token budgeting and semantic compression** (M2-02). Not M1: nothing is saved to disk; the history is summarised in memory. Proof: `test_compression.py::TestSemanticCompression::test_compress_history_preserves_base_system_and_recent_turns`.
4. **X3 Export** (X3-01). Not T5: the quiz already exists; the story is about the files produced. Proof: `test_quiz_export.py::TestQuizCommand::test_student_copy`.
5. **T1 MCP**, custom `teacher` server (T1-01). Not T5: the model calls an external tool and continues with its result. Proof: `test_teacher_tools_server.py::TestTeacherToolsWithSensai::test_tool_call_through_the_registry`.
