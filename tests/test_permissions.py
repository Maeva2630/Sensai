import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pyrefly: ignore [missing-import]
from cli.cli import CliDependencies, cli
# pyrefly: ignore [missing-import]
from mcp.manager import MCPManager
# pyrefly: ignore [missing-import]
from support import use_temporary_working_directory_for_module
# pyrefly: ignore [missing-import]
from tools.base import ToolError
# pyrefly: ignore [missing-import]
from tools.permissions import (
    DEFAULT_PERMISSIONS_PATH,
    AllowedDirectory,
    PermissionChecker,
    PermissionDeniedError,
    load_permissions,
)


def setUpModule():
    # The CLI would start the servers declared in the repository's mcp_servers.json.
    patcher = patch("cli.cli.MCPManager.from_config_file", return_value=MCPManager([]))
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    # cli() saves sessions/, profile.json and data/ in the current directory: not in the repository.
    use_temporary_working_directory_for_module()


class PermissionTestCase(unittest.TestCase):
    """A project folder with a read-only `courses`, a writable `exports` and a `secret` folder."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        for name in ("courses", "exports", "secret"):
            (self.root / name).mkdir()
        (self.root / "courses" / "notes.md").write_text("course", encoding="utf-8")
        (self.root / "secret" / "passwords.txt").write_text("secret", encoding="utf-8")
        self.checker = PermissionChecker(
            [AllowedDirectory(self.root / "courses", ("read",)),
             AllowedDirectory(self.root / "exports", ("read", "write"))],
            deny=[".env", "*.pem"],
            base_dir=self.root,
        )


class TestPermissionChecker(PermissionTestCase):

    def test_read_in_an_allowed_directory(self):
        decision = self.checker.check("courses/notes.md", "read")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.path, self.root / "courses" / "notes.md")

    def test_absolute_path_inside_an_allowed_directory(self):
        self.assertTrue(self.checker.check(str(self.root / "exports" / "quiz.json"), "write").allowed)

    def test_a_file_that_does_not_exist_yet_can_be_written(self):
        self.assertTrue(self.checker.check("exports/new/deep/quiz.json", "write").allowed)

    def test_read_only_directory_refuses_writes(self):
        decision = self.checker.check("courses/notes.md", "write")
        self.assertFalse(decision.allowed)
        self.assertIn("does not allow write access", decision.reason)

    def test_everything_else_is_denied_by_default(self):
        for path in ["secret/passwords.txt", "notes.txt", str(self.root), "/etc/passwd", "/"]:
            with self.subTest(path=path):
                decision = self.checker.check(path, "read")
                self.assertFalse(decision.allowed)
                self.assertIn("outside the allowed directories", decision.reason)

    def test_the_allowed_directory_name_is_not_a_prefix_match(self):
        (self.root / "courses-private").mkdir()
        self.assertFalse(self.checker.check("courses-private/a.txt", "read").allowed)

    def test_path_traversal_is_blocked(self):
        for path in ["../outside.txt", "courses/../secret/passwords.txt", "exports/../../etc/passwd",
                     "courses/../../" + self.root.name + "/secret/passwords.txt"]:
            with self.subTest(path=path):
                self.assertFalse(self.checker.check(path, "read").allowed)

    def test_traversal_that_stays_inside_is_fine(self):
        self.assertTrue(self.checker.check("courses/../exports/a.json", "write").allowed)

    def test_symlink_escaping_an_allowed_directory_is_blocked(self):
        os.symlink(self.root / "secret" / "passwords.txt", self.root / "courses" / "link.txt")
        os.symlink(self.root / "secret", self.root / "exports" / "linked_dir")

        self.assertFalse(self.checker.check("courses/link.txt", "read").allowed)
        self.assertFalse(self.checker.check("exports/linked_dir/passwords.txt", "read").allowed)
        self.assertFalse(self.checker.check("exports/linked_dir/new.txt", "write").allowed)

    def test_symlink_pointing_inside_is_allowed(self):
        os.symlink(self.root / "courses" / "notes.md", self.root / "exports" / "copy.md")
        decision = self.checker.check("exports/copy.md", "read")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.path, self.root / "courses" / "notes.md")

    def test_symlink_loop_does_not_crash_the_check(self):
        os.symlink(self.root / "exports" / "b", self.root / "exports" / "a")
        os.symlink(self.root / "exports" / "a", self.root / "exports" / "b")
        # Depending on the Python version the loop is denied or stays inside exports; opening it fails later
        decision = self.checker.check("exports/a", "read")
        if decision.allowed:
            self.assertTrue(decision.path.is_relative_to(self.root / "exports"))

    def test_home_shortcut_is_expanded_before_the_check(self):
        self.assertFalse(self.checker.check("~/courses/notes.md", "read").allowed)

    def test_deny_rules_win_over_allow(self):
        for path in ["exports/.env", "exports/server.pem", "exports/SERVER.PEM", "exports/sub/.env"]:
            with self.subTest(path=path):
                decision = self.checker.check(path, "write")
                self.assertFalse(decision.allowed)
                self.assertIn("deny rule", decision.reason)

    def test_deny_rules_ignore_folders_above_the_allowed_directory(self):
        hidden = self.root / "project.pem"
        hidden.mkdir()
        checker = PermissionChecker([AllowedDirectory(hidden, ("read",))], deny=["*.pem"], base_dir=self.root)
        self.assertTrue(checker.check("project.pem/notes.md", "read").allowed)

    def test_nested_rule_overrides_its_parent(self):
        (self.root / "exports" / "final").mkdir()
        checker = PermissionChecker(
            [AllowedDirectory(self.root / "exports", ("read", "write")),
             AllowedDirectory(self.root / "exports" / "final", ("read",))],
            base_dir=self.root,
        )
        self.assertTrue(checker.check("exports/draft.json", "write").allowed)
        self.assertFalse(checker.check("exports/final/quiz.json", "write").allowed)
        self.assertTrue(checker.check("exports/final/quiz.json", "read").allowed)

    def test_relative_paths_are_relative_to_the_project_not_the_current_directory(self):
        elsewhere = tempfile.TemporaryDirectory()
        self.addCleanup(elsewhere.cleanup)
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(elsewhere.name)
        self.assertTrue(self.checker.check("courses/notes.md", "read").allowed)

    def test_invalid_input_is_denied_without_crashing(self):
        for path in ["", "   ", None, 42, "courses/\0notes.md", ["courses"]]:
            with self.subTest(path=path):
                self.assertFalse(self.checker.check(path, "read").allowed)

    def test_unknown_mode_is_denied(self):
        decision = self.checker.check("courses/notes.md", "execute")
        self.assertFalse(decision.allowed)
        self.assertIn("unknown access mode", decision.reason)

    def test_no_rules_means_no_access(self):
        self.assertFalse(PermissionChecker(base_dir=self.root).check("courses/notes.md", "read").allowed)

    def test_require_returns_the_real_path_or_raises(self):
        self.assertEqual(self.checker.require("courses/notes.md", "read"), self.root / "courses" / "notes.md")
        with self.assertRaises(PermissionDeniedError) as raised:
            self.checker.require("secret/passwords.txt", "read")
        self.assertIsInstance(raised.exception, ToolError)
        self.assertTrue(str(raised.exception).startswith("Access denied:"))
        self.assertIn("courses (read)", str(raised.exception))

    def test_describe_lists_rules_with_relative_paths(self):
        self.assertEqual(
            self.checker.describe(),
            ["courses: read", "exports: read, write", "deny: .env", "deny: *.pem"],
        )


class TestLoadPermissions(PermissionTestCase):

    def write(self, data):
        path = self.root / "permissions.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def test_loads_directories_and_deny_rules(self):
        checker, warnings = load_permissions(self.write({
            "allowed": [{"path": "courses", "modes": ["read"]}, {"path": "exports", "modes": ["read", "write", "read"]}],
            "deny": [".env"],
        }))

        self.assertEqual(warnings, [])
        self.assertEqual(checker.describe(), ["courses: read", "exports: read, write", "deny: .env"])
        self.assertTrue(checker.check("exports/a.json", "write").allowed)
        self.assertFalse(checker.check("courses/a.md", "write").allowed)

    def test_missing_file_denies_everything(self):
        checker, warnings = load_permissions(self.root / "nope.json")
        self.assertEqual(warnings, [])
        self.assertFalse(checker.check("courses/notes.md", "read").allowed)

    def test_unreadable_file_denies_everything_and_warns(self):
        for content in ["{not json", "[1, 2]"]:
            with self.subTest(content=content):
                checker, warnings = load_permissions(self.write(content))
                self.assertEqual(len(warnings), 1)
                self.assertIn("All file access is denied", warnings[0])
                self.assertFalse(checker.check("courses/notes.md", "read").allowed)

    def test_invalid_entries_are_skipped_with_a_warning(self):
        checker, warnings = load_permissions(self.write({"allowed": [
            "courses",
            {"modes": ["read"]},
            {"path": "courses", "modes": []},
            {"path": "courses", "modes": ["execute"]},
            {"path": "courses", "modes": "read"},
            {"path": "exports", "modes": ["read"]},
        ]}))

        self.assertEqual(len(warnings), 5)
        self.assertEqual(checker.describe(), ["exports: read"])
        self.assertFalse(checker.check("courses/notes.md", "read").allowed)

    def test_invalid_allowed_section_allows_nothing(self):
        checker, warnings = load_permissions(self.write({"allowed": {"path": "courses"}}))
        self.assertEqual(len(warnings), 1)
        self.assertEqual(checker.describe(), [])

    def test_invalid_deny_section_is_reported(self):
        _, warnings = load_permissions(self.write({"allowed": [], "deny": "*.pem"}))
        self.assertEqual(len(warnings), 1)
        self.assertIn("'deny'", warnings[0])

    def test_relative_directories_follow_the_file_location(self):
        previous = os.getcwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(tempfile.gettempdir())
        checker, _ = load_permissions(self.write({"allowed": [{"path": "courses", "modes": ["read"]}]}))
        self.assertTrue(checker.check(str(self.root / "courses" / "notes.md"), "read").allowed)

    def test_repository_permissions_file(self):
        checker, warnings = load_permissions(DEFAULT_PERMISSIONS_PATH)
        self.assertEqual(warnings, [])
        self.assertTrue(checker.check("courses/cours.md", "read").allowed)
        self.assertFalse(checker.check("courses/cours.md", "write").allowed)
        self.assertTrue(checker.check("exports/quiz.json", "write").allowed)
        self.assertFalse(checker.check("source/main.py", "read").allowed)
        self.assertFalse(checker.check("exports/.env", "read").allowed)


class TestPermissionsCommand(unittest.TestCase):

    def run_cli(self, inputs):
        output = io.StringIO()
        dependencies = CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock())
        with patch("cli.cli.OllamaHTTPClient"), patch("builtins.input", side_effect=[*inputs, "exit"]), \
                patch("sys.stdout", output):
            cli(dependencies=dependencies)
        return output.getvalue()

    def test_permissions_command_lists_the_rules(self):
        output = self.run_cli(["/permissions"])
        self.assertIn("File access rules:", output)
        self.assertIn("courses: read", output)
        self.assertIn("exports: read, write", output)

    def test_permissions_command_is_listed_in_help(self):
        self.assertIn("/permissions", self.run_cli(["/help"]))

    def test_permission_warnings_are_displayed_at_startup(self):
        output = io.StringIO()
        broken = (PermissionChecker(base_dir="."), ["Could not read permissions 'x': boom. All file access is denied."])
        with patch("cli.cli.OllamaHTTPClient"), patch("cli.cli.load_permissions", return_value=broken), \
                patch("builtins.input", side_effect=["/permissions", "exit"]), patch("sys.stdout", output):
            cli(dependencies=CliDependencies(rag_context_provider=MagicMock(), rag_indexer=MagicMock()))
        self.assertIn("Warning: Could not read permissions 'x'", output.getvalue())
        self.assertIn("No file access is allowed.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
