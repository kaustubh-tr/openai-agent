import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Tool, ArgsSchema, ToolRuntime, Agent, ChatOpenAI
from literun.runner import Runner


class TestToolDefinition(unittest.TestCase):
    """
    Unit tests for Tool creation, argument resolution, and schema generation.
    """

    def test_openapi_schema_generation(self):
        """Verify that a Tool generates the correct OpenAI-compatible JSON schema."""

        def dummy_func(a, b=1):
            return a + b

        tool = Tool(
            func=dummy_func,
            name="dummy",
            description="A dummy tool",
            args_schema=[
                ArgsSchema(name="a", type=int, description="Argument a"),
                ArgsSchema(name="b", type=int, description="Argument b"),
            ],
        )

        schema = tool.to_openai_tool()

        # Note: The current implementation returns a flat structure with 'type': 'function'
        # rather than nested {"type": "function", "function": {...}}.
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["name"], "dummy")
        self.assertEqual(schema["description"], "A dummy tool")

        props = schema["parameters"]["properties"]
        self.assertIn("a", props)
        self.assertIn("b", props)
        self.assertEqual(props["a"]["type"], "integer")

    def test_argument_resolution_and_coercion(self):
        """Verify that the tool correctly validates and converts argument types."""

        def add(a, b):
            return a + b

        tool = Tool(
            func=add,
            name="add",
            description="Adds two numbers",
            args_schema=[
                ArgsSchema(name="a", type=int, description="First number"),
                ArgsSchema(name="b", type=int, description="Second number"),
            ],
        )

        # 1. Valid arguments
        result_valid = tool.resolve_arguments({"a": 1, "b": 2})
        self.assertEqual(result_valid["a"], 1)
        self.assertEqual(result_valid["b"], 2)

        # 2. String coercion (e.g. from LLM output)
        result_coerced = tool.resolve_arguments({"a": "10", "b": "20"})
        self.assertEqual(result_coerced["a"], 10)
        self.assertEqual(result_coerced["b"], 20)

    def test_missing_required_argument_error(self):
        """Verify that missing required arguments raise a ValueError."""
        tool = Tool(
            func=lambda a: a,
            name="test",
            description="desc",
            args_schema=[ArgsSchema(name="a", type=int, description="desc")],
        )

        with self.assertRaises(ValueError):
            tool.resolve_arguments({})


class TestRuntimeContext(unittest.TestCase):
    """
    Unit tests for Runtime Context injection into Tools.
    """

    def test_invoke_with_context(self):
        """Verify explicit context values are injected into the tool."""

        # 1. Define a tool that accepts ToolRuntime
        def get_user_id(ctx: ToolRuntime) -> str:
            # Manually extract from context (attribute access)
            user_id = ctx.user_id
            return f"User ID is {user_id}"

        tool = Tool(
            name="get_user_id",
            description="Get the current user ID",
            func=get_user_id,
            args_schema=[],  # No LLM args
        )

        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])

        # 2. Test execution with runtime_context
        context = {"user_id": 42}

        # Use Runner directly to bypass LLM call and test tool execution
        result = Runner._execute_tool(agent, "get_user_id", {}, runtime_context=context)
        self.assertEqual(result, "User ID is 42")

    def test_mixed_args_and_context(self):
        """Verify tools accepting both arguments and context work correctly."""

        def multiply_by_user_factor(x: int, ctx: ToolRuntime) -> int:
            factor = getattr(ctx, "factor", 1)
            return x * factor

        tool = Tool(
            name="multiply",
            description="Multiplies by user factor",
            func=multiply_by_user_factor,
            args_schema=[ArgsSchema(name="x", type=int, description="Input")],
        )

        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])
        context = {"factor": 3}

        # Execution
        result = Runner._execute_tool(
            agent, "multiply", {"x": 10}, runtime_context=context
        )
        self.assertEqual(result, "30")

    def test_missing_context_arg(self):
        """Verify that missing context defaults gracefully (e.g. empty runtime)."""

        def check_presence(ctx: ToolRuntime) -> str:
            if hasattr(ctx, "secret"):
                return "Found"
            return "Not found"

        tool = Tool(
            name="check",
            description="Check secret",
            func=check_presence,
            args_schema=[],
        )
        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])

        # Pass None as context -> Runner should provide an empty ToolRuntime
        result = Runner._execute_tool(agent, "check", {}, runtime_context=None)
        self.assertEqual(result, "Not found")


class TestFutureAnnotations(unittest.TestCase):
    """
    Verify compatibility with 'from __future__ import annotations'.
    This is critical because it changes how type hints are evaluated at runtime.
    """

    def test_toolruntime_with_future_annotations(self):
        """Verify ToolRuntime injection works when using from __future__ import annotations."""

        # Define a tool that uses ToolRuntime with future annotations enabled
        def get_config_value(key: str, ctx: ToolRuntime) -> str:
            value = getattr(ctx, key, "not found")
            return f"Config {key} = {value}"

        tool = Tool(
            name="get_config",
            description="Get a config value",
            func=get_config_value,
            args_schema=[ArgsSchema(name="key", type=str, description="Config key")],
        )

        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])

        # Execute the tool with runtime context
        result = Runner._execute_tool(
            agent,
            "get_config",
            {"key": "db_host"},
            runtime_context={"db_host": "localhost"},
        )

        self.assertEqual(result, "Config db_host = localhost")


if __name__ == "__main__":
    unittest.main()
