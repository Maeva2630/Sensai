import json
from pathlib import Path


class ProfileManager:
    """Manage user profile information."""

    DEFAULT_PROFILE = {
        "user_name": "User",
        "preferred_language": "English",
        "domain_expertise": "General",
        "constraints": []
    }

    def __init__(self, profile_path: str | Path | None = None):
        self.profile_path = Path(profile_path) if profile_path else Path("profile.json")
        self.data = self.load()

    def load(self) -> dict:
        """Load profile or create a default profile if it doesn't exist."""
        if not self.profile_path.exists():
            self.save(self.DEFAULT_PROFILE)
            return self.DEFAULT_PROFILE

        with open(self.profile_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, data: dict) -> None:
        """Save the profile to a JSON file."""
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.data = data

    def build_system_prompt(self, base_instructions: str = "You are a helpful assistant.") -> str:
        """Inject profile information as system instructions for LLM."""
        constraints_str = "\n".join(f"- {c}" for c in self.data.get("constraints", [])) or "None"

        return f"""{base_instructions}

USER PROFILE:
- Name: {self.data.get('user_name')}
- Preferred Language: {self.data.get('preferred_language')}
- Domain Expertise: {self.data.get('domain_expertise')}
- Constraints:
{constraints_str}"""