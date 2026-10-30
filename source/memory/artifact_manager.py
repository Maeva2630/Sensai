"""Artifact and State Document Manager.

Manages a living deliverable document that persists and updates incrementally
across conversation turns, distinct from raw chat history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Literal


@dataclass
class Artifact:
    """Represents a structured living state document."""

    title: str
    sections: dict[str, str] = field(default_factory=dict)
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def render_markdown(self) -> str:
        """Render the artifact as standard Markdown."""
        lines: list[str] = [f"# {self.title}", ""]
        for section_title, content in self.sections.items():
            lines.append(f"## {section_title}")
            lines.append("")
            if content.strip():
                lines.append(content.strip())
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def get_outline(self) -> list[str]:
        """Return the list of section titles."""
        return list(self.sections.keys())

    def to_dict(self) -> dict:
        """Serialize artifact metadata and sections to a dictionary."""
        return {
            "title": self.title,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sections": self.sections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Artifact:
        """Deserialize an Artifact instance from a dictionary."""
        return cls(
            title=data.get("title", "Untitled Document"),
            sections=data.get("sections", {}),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class ArtifactManager:
    """Core manager for creating, editing, and persisting living state documents."""

    def __init__(self, session_id: str | None = None, storage_dir: str = "sessions"):
        self.session_id = session_id or "default"
        self.storage_dir = Path(storage_dir)
        self.artifact: Artifact | None = None

    def has_artifact(self) -> bool:
        """Check if an artifact is currently initialized."""
        return self.artifact is not None

    def init_document(self, title: str, template: str = "") -> Artifact:
        """Initialize a new artifact document.
        
        If a template containing Markdown headers (##) is provided,
        it automatically parses and extracts sections.
        """
        sections: dict[str, str] = {}
        if template.strip():
            parsed_title, parsed_sections = self._parse_markdown(template)
            if parsed_sections:
                sections = parsed_sections
            if parsed_title and (not title or title == "Untitled Document"):
                title = parsed_title
            elif parsed_title:
                # If template has an explicit # Title, prefer the template's title
                title = parsed_title

        self.artifact = Artifact(
            title=title or "Untitled Document",
            sections=sections,
            version=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        return self.artifact

    def update_section(
        self,
        section_name: str,
        content: str,
        mode: Literal["replace", "append", "prepend"] = "replace",
    ) -> bool:
        """Update an existing section or create it if missing.

        Modes:
            - 'replace': Overwrites section content completely.
            - 'append': Appends content to the existing section.
            - 'prepend': Prepends content before the existing section.
        """
        if not self.artifact:
            self.init_document(title="Working Document")

        assert self.artifact is not None

        clean_section = section_name.strip().lstrip("#").strip()
        clean_content = content.strip()

        if clean_section not in self.artifact.sections or mode == "replace":
            self.artifact.sections[clean_section] = clean_content
        elif mode == "append":
            existing = self.artifact.sections[clean_section].strip()
            if existing:
                self.artifact.sections[clean_section] = f"{existing}\n\n{clean_content}"
            else:
                self.artifact.sections[clean_section] = clean_content
        elif mode == "prepend":
            existing = self.artifact.sections[clean_section].strip()
            if existing:
                self.artifact.sections[clean_section] = f"{clean_content}\n\n{existing}"
            else:
                self.artifact.sections[clean_section] = clean_content

        self.artifact.version += 1
        self.artifact.updated_at = datetime.now().isoformat()
        return True

    def replace_text(self, target: str, replacement: str) -> bool:
        """Find and replace a specific text snippet across the whole document.

        Returns True if a replacement was made, False otherwise.
        """
        if not self.artifact or not target:
            return False

        found = False

        # Also allow target replacement in title if matched
        if target in self.artifact.title:
            self.artifact.title = self.artifact.title.replace(target, replacement, 1)
            found = True

        new_sections: dict[str, str] = {}
        for section_name, section_content in self.artifact.sections.items():
            new_sec_name = section_name
            if target in section_name:
                new_sec_name = section_name.replace(target, replacement)
                found = True

            new_sec_content = section_content
            if target in section_content:
                new_sec_content = section_content.replace(target, replacement)
                found = True

            new_sections[new_sec_name] = new_sec_content

        if found:
            self.artifact.sections = new_sections
            self.artifact.version += 1
            self.artifact.updated_at = datetime.now().isoformat()

        return found

    def delete_section(self, section_name: str) -> bool:
        """Delete a section by name."""
        if not self.artifact:
            return False

        clean_section = section_name.strip().lstrip("#").strip()
        if clean_section in self.artifact.sections:
            del self.artifact.sections[clean_section]
            self.artifact.version += 1
            self.artifact.updated_at = datetime.now().isoformat()
            return True
        return False

    def get_content(self) -> str:
        """Return the fully rendered Markdown document."""
        if not self.artifact:
            return ""
        return self.artifact.render_markdown()

    def get_outline(self) -> list[str]:
        """Return the list of section headings."""
        if not self.artifact:
            return []
        return self.artifact.get_outline()

    def get_summary(self) -> str:
        """Return a compact outline summary of the artifact."""
        if not self.artifact:
            return "No active artifact."
        sections_str = ", ".join(f"[{s}]" for s in self.get_outline()) or "None"
        return f"Artifact: '{self.artifact.title}' (v{self.artifact.version}) | Sections: {sections_str}"

    def reset(self) -> None:
        """Reset and discard the current artifact."""
        self.artifact = None

    def save(self, filepath: str | Path | None = None) -> Path:
        """Persist the artifact as Markdown and JSON metadata."""
        if not self.artifact:
            raise ValueError("Cannot save: No active artifact to persist.")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if filepath:
            md_path = Path(filepath)
        else:
            md_path = self.storage_dir / f"{self.session_id}_artifact.md"

        json_path = md_path.with_suffix(".json")

        # Write markdown content
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.get_content())

        # Write metadata JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.artifact.to_dict(), f, indent=2, ensure_ascii=False)

        return md_path

    def load(self, filepath: str | Path | None = None) -> Artifact:
        """Load an artifact from disk (JSON metadata preferred, falls back to Markdown)."""
        if filepath:
            target_path = Path(filepath)
        else:
            target_path = self.storage_dir / f"{self.session_id}_artifact.json"

        if target_path.suffix == ".json" and target_path.exists():
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.artifact = Artifact.from_dict(data)
            return self.artifact

        md_path = target_path.with_suffix(".md")
        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                raw_md = f.read()
            title, sections = self._parse_markdown(raw_md)
            self.artifact = Artifact(
                title=title or "Loaded Document",
                sections=sections,
                version=1,
            )
            return self.artifact

        raise FileNotFoundError(f"Artifact not found at {target_path} or {md_path}")

    @staticmethod
    def _parse_markdown(markdown_text: str) -> tuple[str, dict[str, str]]:
        """Parse raw markdown text into document title and a sections dict."""
        lines = markdown_text.splitlines()
        title = "Untitled Document"
        sections: dict[str, str] = {}
        current_section: str | None = None
        current_buffer: list[str] = []

        for line in lines:
            h1_match = re.match(r"^#\s+(.+)$", line)
            h2_match = re.match(r"^##\s+(.+)$", line)

            if h1_match and not current_section and title == "Untitled Document":
                title = h1_match.group(1).strip()
            elif h2_match:
                if current_section:
                    sections[current_section] = "\n".join(current_buffer).strip()
                current_section = h2_match.group(1).strip()
                current_buffer = []
            else:
                if current_section:
                    current_buffer.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_buffer).strip()

        return title, sections
