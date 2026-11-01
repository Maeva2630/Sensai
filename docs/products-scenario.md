# Product scenario and traceability

[Home](../README.md) · [User guide](user-guide.md) · [Feature documentation](features/README.md)

## Product vision

SensAI is a local assistant for teachers preparing learning activities, classroom materials, and assessments. It consults local documents, adapts activities, generates quizzes, and checks routine calculations while the teacher retains control of the final material and local data. Generated answers, sources, grades, and exports must always be reviewed before use with learners.

## Target users

| Audience | Available SensAI capabilities | Product positioning |
| --- | --- | --- |
| Primary-school teachers | All implemented features: RAG, tools and MCP, personas, sessions, logging, and monitoring. Typical uses include clear activities, local resources, short quizzes, and age-appropriate explanations. | The teacher checks suitability for pupils and validates generated content. |
| Middle-school teachers | All implemented features: RAG, tools and MCP, personas, sessions, logging, and monitoring. Typical uses include subject resources, assessments, grade calculations, timers, and longer preparation sessions. | SensAI only knows a curriculum supplied in local documents. |
| High-school teachers | All implemented features are technically available in the same application. | The current model choice and product scenario are designed and validated for primary and middle-school use; high-school pedagogical use requires further evaluation, not new feature access. |

## Final demonstration scenario

The demonstration follows **Lucas**, a middle-school mathematics teacher preparing a fractions lesson for a Year 6 / *sixième* class. Fractions also fit a primary-school transition activity, so the workflow represents both current target audiences.

Before the demonstration, prepare a fictional UTF-8 course file such as `courses/fractions.md`, a non-sensitive `profile.json`, and a running Ollama server.

1. Start SensAI and show `/help`.
2. Select `/persona use mathematics-assistant`.
3. Run `/ingest courses/fractions.md`, `/rag on`, then ask a course question.
4. Use `/rag filter name fractions.md` and `/sources on` or `/sources off`.
5. Run `/quiz --questions 3 Les fractions` and optionally save it to `exports/`.
6. Ask for a weighted-grade calculation or a short timer, demonstrating a teacher MCP tool.
7. Run `/save`, `/tokens`, and, with a prepared long session, `/compress`.
8. Run `/debug prompt`; with monitoring enabled beforehand, run `/metrics`.

Use only fictional or public teaching material during the demonstration. Never enter real pupil names, grades, or personal data.

## User stories and acceptance criteria

### RAG

**RAG-01 - Use a local teaching resource**

As a primary or middle-school teacher, I want to index a local course and ask a question about it so that I can prepare material from a source I control.

- `/ingest` accepts supported text, source-code, PDF, DOCX, and ZIP files.
- With `/rag on`, relevant indexed excerpts are sent with the question.
- Source filename and chunk identifiers can be displayed after a grounded answer.
- Missing, empty, unsupported, or scanned PDF files fail with a readable message.

**RAG-02 - Limit research to the right source**

As a middle-school teacher, I want to restrict retrieval to selected resources so that a worksheet is not mixed with unrelated material.

- `/rag filter name <file>` and `/rag filter category <type>` limit candidate chunks.
- `/rag filter clear` restores normal retrieval.
- `/sources off` hides attribution only; it does not disable RAG.

### Tools and MCP

**TOOLS-01 - Prepare an assessment efficiently**

As a teacher, I want to generate and save a short course-grounded quiz so that I can prepare an assessment more quickly.

- `/quiz --questions <1-20> <topic>` produces the requested number of validated questions.
- With RAG enabled and no matching course excerpt, generation stops with a clear message.
- `/quiz --save <path> --no-answers` creates a learner copy only in a permitted directory.
- Existing exports are kept unless `--overwrite` is explicitly supplied.

**TOOLS-02 - Use controlled classroom utilities**

As a teacher, I want to calculate grades, inspect class statistics, or start an exercise timer so that routine preparation is faster.

- `/tools` lists available native and MCP tools with their origin.
- The included teacher MCP server provides prefixed grade, statistics, and timer tools.
- Invalid tool input or an unavailable MCP server does not terminate SensAI.
- File operations obey `permissions.json`; denied paths and traversal attempts are refused.

### Personas

**PERSONA-01 - Select a subject-appropriate assistant**

As a teacher, I want to choose a persona for my preparation task so that the assistant's tone and scope are appropriate.

- `/persona list` identifies available personas and the active one.
- `/persona status` displays safe metadata for the active persona.
- `/persona use <id>` changes persona, rejects an unknown identifier, and keeps history.
- The following model request uses the selected persona instructions.

**PERSONA-02 - Add a persona through configuration**

As a maintainer, I want to add a persona without Python changes so that new subjects can be supported safely.

- Each persona has an identifier, display name, role, tone, scope, and one prompt source.
- The source is either `system_prompt` or a UTF-8 `prompt_file` under the config directory.
- Invalid persona configuration is reported at startup.
- The runtime selection resets on restart unless `personas.active_persona` is changed.

### Sessions, user profile, and conversation memory

**MEMORY-01 - Resume preparation later**

As a teacher, I want to save and reload a preparation conversation so that I can continue working after closing SensAI.

- `/save` writes `sessions/<session_id>.json` when memory is enabled.
- `exit`, `quit`, `Ctrl+C`, and `Ctrl+D` save the current session when memory is enabled.
- `/load <session_id>` restores prior user and assistant history while preserving the system prompt.
- A missing session is reported without ending the CLI.

**MEMORY-02 - Manage a long conversation**

As a teacher, I want to see and manage context use so that a long preparation does not exhaust the local model window unexpectedly.

- `/tokens` displays last-turn, context, and cumulative token estimates.
- A warning appears at 80% of `model.context_window`.
- `/compress` preserves base instructions and recent dialogue while summarizing older turns.
- Automatic compression can run at 85%; this is conversation memory, not long-term pupil profiling.

### Logging and monitoring

**OBS-01 - Diagnose a surprising response**

As a developer, I want local logs and request inspection so that I can understand RAG, persona, and tool behaviour without temporary `print` statements.

- Logs contain timestamp, level, and component in the configured local file.
- Normal chat does not print debug logs.
- `/debug prompt` labels user, assistant, RAG, and tool messages while hiding system/profile data.
- `/debug prompt --include-system` exposes the complete request only on demand.

**OBS-02 - Observe local reliability**

As a developer, I want local metrics so that I can identify slow calls and errors after a model, prompt, or feature change.

- Monitoring only records events when `monitoring.enabled` is true.
- `/metrics` reports volume, latency, errors, token estimates, RAG, tools, and daily calls.
- Date, model, prompt version, feature, tool, and error filters combine with AND semantics.
- Raw prompts, responses, excerpts, tool arguments, and profile data are not stored by default.

### CLI configuration and error handling

**CLI-01 - Configure a local installation**

As a teacher or maintainer, I want one configuration file so that SensAI can be adapted to the local computer without code changes.

- `config.json` controls models, Ollama URL, temperature, context, features, RAG, logs, monitoring, and personas.
- Invalid required values are rejected at startup with a named configuration error.
- `temperature` accepts 0 to 2 and `context_window` accepts integers of at least 512.
- Changes apply after restarting SensAI.

**CLI-02 - Recover from a failure clearly**

As a teacher, I want concise errors so that a local failure does not expose a traceback or prevent me from continuing.

- Ollama, RAG, file, session, and unexpected errors are handled at the CLI boundary.
- Recoverable failures do not show a full traceback during normal use.
- The following user input can still be processed after a recoverable error.
- Technical diagnostics remain in the local log file.

## Traceability table

| Story | Feature | Command or configuration | Automated test | Live demo |
| --- | --- | --- | --- | --- |
| RAG-01 | RAG | `/ingest`, `/rag on` | `test_rag_integration.py::test_indexed_excerpt_is_injected_and_attributed_without_ollama` | Yes, step 3 |
| RAG-02 | RAG | `/rag filter`, `/sources` | `test_cli.py::test_rag_filter_commands_set_and_clear_metadata_restrictions` | Yes, step 4 |
| TOOLS-01 | Quiz and export | `/quiz`, `permissions.json` | `test_structured_output.py::test_quiz_is_built_from_the_course_when_rag_is_on`; `test_quiz_export.py::test_student_copy` | Yes, step 5 |
| TOOLS-02 | Tools and MCP | `/tools`, `/permissions`, `mcp_servers.json` | `test_mcp_manager.py::test_broken_server_does_not_prevent_the_others`; `test_permissions.py::test_path_traversal_is_blocked` | Yes, step 6 |
| PERSONA-01 | Personas | `/persona list`, `/persona status`, `/persona use` | `test_cli.py::test_persona_commands_list_status_and_switch_without_erasing_history` | Yes, step 2 |
| PERSONA-02 | Personas | `personas.profiles`, `prompt_file` | `test_config.py::test_loads_a_persona_prompt_from_a_file_relative_to_the_configuration` | Setup |
| MEMORY-01 | Sessions and memory | `/save`, `/load`, `--session` | `test_cli.py::test_slash_load_restores_conversation_context_for_next_turn` | Yes, step 7 |
| MEMORY-02 | Tokens and compression | `/tokens`, `/compress`, `model.context_window` | `test_cli.py::test_auto_compression_triggered_when_budget_critical`; `test_compression.py::test_compress_history_preserves_base_system_and_recent_turns` | Tokens: yes; compression: optional |
| OBS-01 | Logs and prompt inspection | `logging`, `/debug prompt` | `test_observability.py::test_debug_prompt_displays_user_assistant_and_tool_messages` | Yes, step 8 |
| OBS-02 | Monitoring | `monitoring.enabled`, `/metrics` | `test_monitoring.py::test_persists_aggregates_and_filters_safe_interaction_metadata` | Yes, step 8 |
| CLI-01 | Configuration | `config.json` | `test_config.py::test_rejects_invalid_values_before_startup` | Setup |
| CLI-02 | Error handling | CLI error boundary | `test_cli.py::test_cli_unexpected_error_is_hidden_and_the_next_turn_still_works` | Optional; automated test is safer |

## Reviewer exercise

Give a reviewer this story without its identifier:

> As a teacher, I want to restore yesterday's preparation discussion after restarting SensAI, so that I do not need to repeat my work.

Expected answer: **Sessions, user profile, and conversation memory**. The reviewer should identify `/save`, `/load <session_id>`, or `--session <session_id>`, then find the **MEMORY-01** test evidence in the table.

## Future compatibility and limits

SensAI exposes the same implemented features to every teacher, including a high-school teacher. The current product scenario, prompts, and chosen local model are nevertheless designed and validated for primary and middle-school teaching, where the expected pedagogical complexity is lower. Future high-school work concerns model evaluation, subject-specific prompts, curriculum documents, assessment workflows, and validation with high-school teachers; it does not mean that RAG, tools, MCP, sessions, or the other existing features are unavailable.
