"""Artifact editing and state management tool.

Allows the model to incrementally initialize, modify, and maintain a living deliverable
document across turns without rewriting the entire document.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from memory.artifact_manager import ArtifactManager
from tools.base import Tool


class ArtifactTool(Tool):
    """Tool allowing an LLM agent to inspect and edit a living state document."""

    name = "edit_artifact"
    description = (
        "Create, update, or inspect the living deliverable artifact document. "
        "Use this tool to incrementally build specifications, summaries, notes, or plans "
        "section-by-section instead of regurgitating whole documents in chat responses."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["init", "update_section", "replace_text", "delete_section", "view"],
                "description": (
                    "The operation to perform: 'init' creates/resets the document; "
                    "'update_section' adds or updates a specific section; "
                    "'replace_text' replaces a target snippet across the document; "
                    "'delete_section' removes a section; 'view' returns current outline/content."
                ),
            },
            "title": {
                "type": "string",
                "description": "Document title (used primarily with 'init').",
            },
            "section_name": {
                "type": "string",
                "description": "The title of the section to add, modify, or delete.",
            },
            "content": {
                "type": "string",
                "description": "Content for the section or template for document initialization.",
            },
            "mode": {
                "type": "string",
                "enum": ["replace", "append", "prepend"],
                "description": "Update mode for 'update_section' (default: 'replace').",
            },
            "target": {
                "type": "string",
                "description": "Target string to find and replace (for 'replace_text').",
            },
            "replacement": {
                "type": "string",
                "description": "Replacement string (for 'replace_text').",
            },
        },
        "required": ["action"],
    }

    _RELEVANT_PATTERN = re.compile(
        r"\b(artifact|document|livrable|spec|specification|cahier des charges|plan|notes|summary|résumé|outline)\b",
        re.IGNORECASE,
    )

    def __init__(self, manager: ArtifactManager | None = None):
        self.manager = manager or ArtifactManager()

    def execute(self, arguments: Dict[str, Any]) -> str:
        """Execute the requested artifact operation."""
        action = arguments.get("action")

        if action == "init":
            title = arguments.get("title") or "Working Document"
            template = arguments.get("content") or ""
            artifact = self.manager.init_document(title=title, template=template)
            return (
                f"Artifact initialized: '{artifact.title}' (version {artifact.version}). "
                f"Sections: {', '.join(artifact.get_outline()) or 'None'}."
            )

        if action == "update_section":
            section_name = arguments.get("section_name")
            if not section_name:
                return "Error: 'section_name' is required for action 'update_section'."
            content = arguments.get("content", "")
            mode = arguments.get("mode", "replace")
            if mode not in ("replace", "append", "prepend"):
                mode = "replace"

            self.manager.update_section(section_name=section_name, content=content, mode=mode)
            assert self.manager.artifact is not None
            return (
                f"Section '{section_name}' updated (mode: {mode}). "
                f"Document '{self.manager.artifact.title}' is now at version {self.manager.artifact.version}."
            )

        if action == "replace_text":
            target = arguments.get("target")
            replacement = arguments.get("replacement", "")
            if not target:
                return "Error: 'target' is required for action 'replace_text'."

            success = self.manager.replace_text(target=target, replacement=replacement)
            if success:
                return f"Replaced occurrences of '{target}' with '{replacement}'. Document version incremented."
            return f"Target string '{target}' not found in artifact."

        if action == "delete_section":
            section_name = arguments.get("section_name")
            if not section_name:
                return "Error: 'section_name' is required for action 'delete_section'."
            success = self.manager.delete_section(section_name)
            if success:
                return f"Section '{section_name}' deleted."
            return f"Section '{section_name}' not found."

        if action == "view":
            if not self.manager.has_artifact():
                return "No active artifact currently initialized."
            return self.manager.get_content()

        return f"Unknown action '{action}'."

    def is_relevant_for(self, user_input: str) -> bool:
        """Available by default if an artifact already exists, or if requested."""
        if self.manager.has_artifact():
            return True
        return bool(self._REQUEST_PATTERN_SEARCH(user_input))

    def _REQUEST_PATTERN_SEARCH(self, text: str) -> bool:
        return bool(self._RELEVANT_PATTERN.search(text))
