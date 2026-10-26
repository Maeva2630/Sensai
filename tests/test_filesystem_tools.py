import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from cli.cli import default_registry
# pyrefly: ignore [missing-import]
from tools.filesystem import (
    MAX_LISTED_ENTRIES,
    MAX_READ_BYTES,
    MAX_WRITE_CHARS,
    CreateDirectoryTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    filesystem_tools,
)
# pyrefly: ignore [missing-import]
from tools.permissions import AllowedDirectory, PermissionChecker
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry


class FilesystemTestCase(unittest.TestCase):
    """A project with a read-only `courses`, a writable `exports` and a `secret` folder."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        for name in ("courses", "exports", "secret"):
            (self.root / name).mkdir()
        (self.root / "courses" / "notes.csv").write_text("eleve,note\nLea,12\n", encoding="utf-8")
        (self.root / "secret" / "passwords.txt").write_text("secret", encoding="utf-8")
        self.checker = PermissionChecker(
            [AllowedDirectory(self.root / "courses", ("read",)),
             AllowedDirectory(self.root / "exports", ("read", "write"))],
            deny=[".env"],
            base_dir=self.root,
        )
        self.registry = ToolRegistry()
        for tool in filesystem_tools(self.checker):
            self.registry.register(tool)

    def run_tool(self, name, **arguments):
        return self.registry.execute(name, arguments)


class TestReadFile(FilesystemTestCase):

    def test_reads_a_file(self):
        result = self.run_tool("read_file", path="courses/notes.csv")
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "eleve,note\nLea,12\n")

    def test_missing_file_and_directory(self):
        self.assertIn("does not exist", self.run_tool("read_file", path="courses/nope.csv").content)
        self.assertIn("is not a file", self.run_tool("read_file", path="courses").content)

    def test_binary_and_non_utf8_files_are_refused(self):
        (self.root / "exports" / "image.bin").write_bytes(b"\x89PNG\0\0data")
        (self.root / "exports" / "latin.txt").write_bytes("café".encode("latin-1"))
        self.assertIn("binary", self.run_tool("read_file", path="exports/image.bin").content)
        self.assertIn("not a UTF-8", self.run_tool("read_file", path="exports/latin.txt").content)

    def test_long_file_is_truncated_with_a_marker(self):
        (self.root / "exports" / "big.txt").write_text("a" * (MAX_READ_BYTES + 500), encoding="utf-8")
        result = self.run_tool("read_file", path="exports/big.txt")
        self.assertFalse(result.is_error)
        self.assertIn("file truncated", result.content)
        self.assertLess(len(result.content), MAX_READ_BYTES + 200)
        self.assertNotIn("omitted", result.content)  # the tool's own marker, not the registry's

    def test_cut_in_the_middle_of_a_character_is_not_an_error(self):
        (self.root / "exports" / "accents.txt").write_text("€" * MAX_READ_BYTES, encoding="utf-8")
        result = self.run_tool("read_file", path="exports/accents.txt")
        self.assertFalse(result.is_error)
        self.assertIn("file truncated", result.content)

    def test_denied_paths(self):
        for path in ["secret/passwords.txt", "../secret/passwords.txt", "/etc/passwd", "courses/../secret/passwords.txt"]:
            with self.subTest(path=path):
                result = self.run_tool("read_file", path=path)
                self.assertTrue(result.is_error)
                self.assertIn("Access denied", result.content)

    def test_denied_path_does_not_reveal_whether_it_exists(self):
        existing = self.run_tool("read_file", path="secret/passwords.txt").content
        missing = self.run_tool("read_file", path="secret/other.txt").content
        self.assertNotIn("does not exist", missing)
        self.assertEqual(existing.replace("passwords.txt", "x"), missing.replace("other.txt", "x"))

    def test_deny_rule_blocks_an_allowed_directory(self):
        (self.root / "exports" / ".env").write_text("KEY=1", encoding="utf-8")
        self.assertIn("deny rule", self.run_tool("read_file", path="exports/.env").content)

    def test_symlink_leaving_the_allowed_directory_is_refused(self):
        os.symlink(self.root / "secret" / "passwords.txt", self.root / "courses" / "link.txt")
        self.assertIn("Access denied", self.run_tool("read_file", path="courses/link.txt").content)

    def test_os_error_is_reported_without_the_absolute_path(self):
        with patch.object(Path, "open", side_effect=PermissionError(13, "Permission denied")):
            result = self.run_tool("read_file", path="courses/notes.csv")
        self.assertTrue(result.is_error)
        self.assertIn("Could not read 'courses/notes.csv': Permission denied", result.content)
        self.assertNotIn(str(self.root), result.content)


class TestWriteFile(FilesystemTestCase):

    def test_writes_a_file(self):
        result = self.run_tool("write_file", path="exports/quiz.md", content="# Quiz")
        self.assertFalse(result.is_error)
        self.assertEqual((self.root / "exports" / "quiz.md").read_text(encoding="utf-8"), "# Quiz")
        self.assertIn("Wrote 6 characters", result.content)

    def test_creates_missing_folders(self):
        self.run_tool("write_file", path="exports/6a/quiz.md", content="x")
        self.assertTrue((self.root / "exports" / "6a" / "quiz.md").exists())

    def test_existing_file_is_kept_unless_overwrite(self):
        self.run_tool("write_file", path="exports/quiz.md", content="first")

        refused = self.run_tool("write_file", path="exports/quiz.md", content="second")
        self.assertTrue(refused.is_error)
        self.assertIn("already exists", refused.content)
        self.assertEqual((self.root / "exports" / "quiz.md").read_text(encoding="utf-8"), "first")

        self.assertFalse(self.run_tool("write_file", path="exports/quiz.md", content="second", overwrite=True).is_error)
        self.assertEqual((self.root / "exports" / "quiz.md").read_text(encoding="utf-8"), "second")

    def test_read_only_and_outside_directories_are_refused(self):
        for path in ["courses/notes.csv", "courses/new.txt", "secret/new.txt", "../new.txt", "exports/../courses/new.txt"]:
            with self.subTest(path=path):
                result = self.run_tool("write_file", path=path, content="x", overwrite=True)
                self.assertTrue(result.is_error)
                self.assertIn("Access denied", result.content)
        self.assertEqual((self.root / "courses" / "notes.csv").read_text(encoding="utf-8"), "eleve,note\nLea,12\n")
        self.assertFalse((self.root / "courses" / "new.txt").exists())

    def test_content_too_long(self):
        result = self.run_tool("write_file", path="exports/big.txt", content="a" * (MAX_WRITE_CHARS + 1))
        self.assertTrue(result.is_error)
        self.assertIn("too long", result.content)
        self.assertFalse((self.root / "exports" / "big.txt").exists())

    def test_target_is_a_directory(self):
        self.assertIn("is a directory", self.run_tool("write_file", path="exports", content="x").content)

    def test_missing_arguments_are_rejected_by_validation(self):
        self.assertIn("missing required argument 'content'", self.run_tool("write_file", path="exports/a.txt").content)

    def test_os_error_is_reported(self):
        with patch.object(Path, "write_text", side_effect=OSError(28, "No space left on device")):
            result = self.run_tool("write_file", path="exports/a.txt", content="x")
        self.assertTrue(result.is_error)
        self.assertIn("No space left on device", result.content)


class TestCreateDirectory(FilesystemTestCase):

    def test_creates_nested_directories_and_is_repeatable(self):
        self.assertFalse(self.run_tool("create_directory", path="exports/6a/quiz").is_error)
        self.assertTrue((self.root / "exports" / "6a" / "quiz").is_dir())
        self.assertFalse(self.run_tool("create_directory", path="exports/6a/quiz").is_error)

    def test_a_file_is_in_the_way(self):
        (self.root / "exports" / "taken").write_text("x", encoding="utf-8")
        self.assertIn("is not a directory", self.run_tool("create_directory", path="exports/taken").content)

    def test_read_only_and_outside_directories_are_refused(self):
        for path in ["courses/new", "secret/new", "../new"]:
            with self.subTest(path=path):
                self.assertIn("Access denied", self.run_tool("create_directory", path=path).content)
        self.assertFalse((self.root / "courses" / "new").exists())


class TestListDirectory(FilesystemTestCase):

    def test_lists_folders_first_then_files(self):
        (self.root / "exports" / "zeta.txt").write_text("x", encoding="utf-8")
        (self.root / "exports" / "alpha.txt").write_text("x", encoding="utf-8")
        (self.root / "exports" / "6a").mkdir()
        self.assertEqual(self.run_tool("list_directory", path="exports").content, "6a/\nalpha.txt\nzeta.txt")

    def test_entries_that_cannot_be_opened_are_hidden(self):
        (self.root / "exports" / "visible.txt").write_text("x", encoding="utf-8")
        (self.root / "exports" / ".env").write_text("KEY=1", encoding="utf-8")
        os.symlink(self.root / "secret" / "passwords.txt", self.root / "exports" / "link.txt")
        self.assertEqual(self.run_tool("list_directory", path="exports").content, "visible.txt")

    def test_empty_missing_and_not_a_directory(self):
        self.assertIn("is empty", self.run_tool("list_directory", path="exports").content)
        self.assertIn("does not exist", self.run_tool("list_directory", path="exports/nope").content)
        self.assertIn("is not a directory", self.run_tool("list_directory", path="courses/notes.csv").content)

    def test_long_listing_is_limited(self):
        for number in range(MAX_LISTED_ENTRIES + 5):
            (self.root / "exports" / f"f{number:04d}.txt").write_text("x", encoding="utf-8")
        lines = self.run_tool("list_directory", path="exports").content.splitlines()
        self.assertEqual(len(lines), MAX_LISTED_ENTRIES + 1)
        self.assertEqual(lines[-1], "... and 5 more")

    def test_denied_directory(self):
        self.assertIn("Access denied", self.run_tool("list_directory", path="secret").content)
        self.assertIn("Access denied", self.run_tool("list_directory", path=".").content)


class TestPermissionCheckComesFirst(FilesystemTestCase):

    def test_no_operation_runs_when_the_permission_check_fails(self):
        calls = [
            (WriteFileTool, {"path": "exports/a.txt", "content": "x"}),
            (CreateDirectoryTool, {"path": "exports/dir"}),
            (ReadFileTool, {"path": "courses/notes.csv"}),
            (ListDirectoryTool, {"path": "courses"}),
        ]
        for tool_class, arguments in calls:
            with self.subTest(tool=tool_class.name):
                tool = tool_class(self.checker)
                with patch.object(self.checker, "require", side_effect=PermissionError("no")), \
                        patch.object(Path, "write_text") as write, patch.object(Path, "mkdir") as mkdir, \
                        patch.object(Path, "open") as open_, patch.object(Path, "iterdir") as iterdir, \
                        patch.object(Path, "exists") as exists:
                    with self.assertRaises(PermissionError):
                        tool.execute(arguments)
                for mocked in (write, mkdir, open_, iterdir, exists):
                    mocked.assert_not_called()


class TestToolRouting(FilesystemTestCase):

    def test_file_tools_are_offered_only_for_file_requests(self):
        for text in ["Lis le fichier courses/notes.csv", "Exporte le quiz", "Write the result in a file",
                     "Enregistre ce résumé", "Liste le dossier courses", "Save this", "Écris un résumé"]:
            with self.subTest(text=text):
                self.assertEqual(len(self.registry.to_ollama_tools(text)), 4)
        for text in ["What is 17 + 25?", "Explique le théorème de Pythagore", "Quelle heure est-il ?", "I already know"]:
            with self.subTest(text=text):
                self.assertEqual(self.registry.to_ollama_tools(text), [])

    def test_file_tools_are_untrusted_and_never_claim_a_request(self):
        for tool in self.registry.list_tools():
            self.assertTrue(tool.untrusted)
            self.assertFalse(tool.claims_request("Lis le fichier courses/notes.csv"))


class TestDefaultRegistry(FilesystemTestCase):

    def test_filesystem_tools_are_registered_with_a_permission_checker(self):
        names = [tool.name for tool in default_registry(True, self.checker).list_tools()]
        self.assertEqual(names, ["get_time", "read_file", "write_file", "create_directory", "list_directory"])

    def test_no_filesystem_tools_without_a_checker_or_when_tools_are_disabled(self):
        self.assertEqual([tool.name for tool in default_registry(True).list_tools()], ["get_time"])
        self.assertEqual(default_registry(False, self.checker).list_tools(), [])


if __name__ == "__main__":
    unittest.main()
