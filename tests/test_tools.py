import unittest
import sys
import os

# Ensure `source` directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source")))

# pyrefly: ignore [missing-import]
from tools.base import Tool
# pyrefly: ignore [missing-import]
from tools.registry import ToolRegistry, validate_arguments
# pyrefly: ignore [missing-import]
from tools.time_tool import GetTimeTool


class AddTool(Tool):
    name = "add"
    description = "Add two integers."
    parameters = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }

    def execute(self, arguments):
        return arguments["a"] + arguments["b"]


class BrokenTool(Tool):
    name = "broken"
    description = "Always fails."

    def execute(self, arguments):
        raise RuntimeError("boom")


class TestValidateArguments(unittest.TestCase):

    def test_valid_arguments(self):
        self.assertIsNone(validate_arguments(AddTool.parameters, {"a": 1, "b": 2}))

    def test_missing_required(self):
        self.assertIn("missing required", validate_arguments(AddTool.parameters, {"a": 1}))

    def test_unknown_argument(self):
        self.assertIn("unknown argument", validate_arguments(AddTool.parameters, {"a": 1, "b": 2, "c": 3}))

    def test_wrong_type(self):
        self.assertIn("type integer", validate_arguments(AddTool.parameters, {"a": "1", "b": 2}))

    def test_bool_is_not_a_number(self):
        self.assertIn("type integer", validate_arguments(AddTool.parameters, {"a": True, "b": 2}))

    def test_enum(self):
        schema = {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}}}
        self.assertIsNone(validate_arguments(schema, {"unit": "c"}))
        self.assertIn("one of", validate_arguments(schema, {"unit": "k"}))

    def test_type_list_accepts_any_listed_type(self):
        schema = {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}
        self.assertIsNone(validate_arguments(schema, {"note": "hi"}))
        self.assertIsNone(validate_arguments(schema, {"note": None}))
        self.assertIn("string or null", validate_arguments(schema, {"note": 3}))

    def test_additional_properties_allowed_when_declared(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": True}
        self.assertIsNone(validate_arguments(schema, {"anything": 1}))

    def test_property_without_type_is_accepted(self):
        schema = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}, "y": True}}
        self.assertIsNone(validate_arguments(schema, {"x": 1, "y": "free"}))

    def test_arguments_must_be_an_object(self):
        self.assertIn("object", validate_arguments(AddTool.parameters, "not a dict"))


class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(AddTool())

    def test_register_and_get(self):
        self.assertIsInstance(self.registry.get("add"), AddTool)
        self.assertIsNone(self.registry.get("missing"))
        self.assertEqual([t.name for t in self.registry.list_tools()], ["add"])

    def test_register_duplicate_raises(self):
        with self.assertRaises(ValueError):
            self.registry.register(AddTool())

    def test_register_without_name_raises(self):
        class Nameless(Tool):
            def execute(self, arguments):
                return ""
        with self.assertRaises(ValueError):
            self.registry.register(Nameless())

    def test_to_ollama_tools(self):
        self.assertEqual(self.registry.to_ollama_tools(), [{
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": AddTool.parameters,
            },
        }])

    def test_execute_success(self):
        result = self.registry.execute("add", {"a": 17, "b": 25})
        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "42")

    def test_execute_unknown_tool(self):
        result = self.registry.execute("missing", {})
        self.assertTrue(result.is_error)
        self.assertIn("Unknown tool", result.content)

    def test_execute_invalid_arguments(self):
        result = self.registry.execute("add", {"a": 1})
        self.assertTrue(result.is_error)
        self.assertIn("Invalid arguments", result.content)

    def test_execute_tool_exception_is_returned_not_raised(self):
        self.registry.register(BrokenTool())
        result = self.registry.execute("broken", {})
        self.assertTrue(result.is_error)
        self.assertIn("boom", result.content)


class TestGetTimeTool(unittest.TestCase):

    def test_get_time(self):
        registry = ToolRegistry()
        registry.register(GetTimeTool())
        result = registry.execute("get_time", {})
        self.assertFalse(result.is_error)
        self.assertRegex(result.content, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main()
