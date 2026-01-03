import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import Tool, Arg


class TestTool(unittest.TestCase):
    def test_tool_schema(self):
        def dummy_func(a, b=1):
            return a + b

        tool = Tool(
            name="dummy",
            description="A dummy tool",
            func=dummy_func,
            args=[
                Arg("a", int, "Argument a"),
                Arg("b", int, "Argument b")
            ]
        )

        schema = tool.to_openai_tool()
        self.assertEqual(schema["name"], "dummy")
        self.assertEqual(schema["description"], "A dummy tool")
        
        props = schema["parameters"]["properties"]
        self.assertIn("a", props)
        self.assertIn("b", props)
        self.assertEqual(props["a"]["type"], "integer")

    def test_argument_resolution(self):
        def add(a, b):
            return a + b

        tool = Tool(
            name="add",
            description="Adds two numbers",
            func=add,
            args=[
                Arg("a", int, "First number"),
                Arg("b", int, "Second number")
            ]
        )

        # Test with valid args
        result = tool.resolve_arguments({"a": 1, "b": 2})
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)

        # Test with string coercion
        result = tool.resolve_arguments({"a": "10", "b": "20"})
        self.assertEqual(result["a"], 10)
        self.assertEqual(result["b"], 20)

    def test_missing_argument(self):
        tool = Tool("test", "desc", [Arg("a", int, "desc")], lambda a: a)
        with self.assertRaises(ValueError):
            tool.resolve_arguments({})

if __name__ == "__main__":
    unittest.main()
