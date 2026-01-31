import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool, ArgsSchema, ChatOpenAI
from literun.results import RunResult

# Check if API key is set to skip integration tests
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None


class TestAgentConstructor(unittest.TestCase):
    """
    Unit tests for Agent initialization and configuration
    that do not require external API calls.
    """

    def test_tools_registration(self):
        """Verify tools are correctly registered during initialization."""
        tool = Tool(
            func=lambda: None,
            name="test_tool",
            description="desc",
            args_schema=[],
        )
        llm = ChatOpenAI(model="gpt-4o", api_key="fake-api-key")  # Mock key
        agent = Agent(llm=llm, tools=[tool])

        # Verify tool is stored in internal list
        self.assertIn(tool, agent.tools)

    def test_duplicate_tools_error(self):
        """Verify that registering duplicate tools raises a ValueError."""
        tool = Tool(
            func=lambda: None,
            name="test_tool",
            description="desc",
            args_schema=[],
        )
        llm = ChatOpenAI(model="gpt-4o", api_key="fake-api-key")

        with self.assertRaises(ValueError):
            Agent(llm=llm, tools=[tool, tool])


@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestAgentIntegration(unittest.TestCase):
    """
    Integration tests for Agent execution using the real OpenAI API.
    WARNING: These tests incur costs.
    """

    def setUp(self):
        """Initialize common checking tools for tests."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    def test_initialization_defaults(self):
        """Verify default values for an Agent instance."""
        agent = Agent(llm=self.llm, system_prompt="Test system prompt")

        self.assertEqual(agent.llm.model, "gpt-4o")
        self.assertEqual(agent.system_prompt, "Test system prompt")
        self.assertIsNone(agent.tools)

    def test_invoke_basic_response(self):
        """Verify a simple synchronous invocation (chat)."""
        agent = Agent(llm=self.llm, system_prompt="You are a helpful assistant.")

        response = agent.invoke(user_input="Say 'Hello world' and nothing else.")

        self.assertIsInstance(response, RunResult)
        self.assertIn("Hello world", response.final_output)

    def test_invoke_with_tool_call(self):
        """Verify the agent can correctly call a tool and return the output."""

        # 1. Define Tool
        def echo(msg: str) -> str:
            return f"Echo: {msg}"

        tool = Tool(
            name="echo",
            description="Echoes the message back to the user",
            args_schema=[
                ArgsSchema(name="msg", type=str, description="The message to echo")
            ],
            func=echo,
        )

        # 2. Setup Agent
        agent = Agent(
            llm=self.llm,
            system_prompt="You are a helpful assistant. Use the echo tool when asked.",
            tools=[tool],
        )

        # 3. Execute
        response = agent.invoke(user_input="Please use the echo tool to say 'hello'")

        # 4. Assert
        self.assertIn("Echo: hello", response.final_output)

    def test_stream_basic_response(self):
        """Verify streaming execution collects text deltas correctly."""
        agent = Agent(llm=self.llm, system_prompt="You are a helpful assistant.")

        chunks = []
        for result in agent.stream(user_input="Say 'Hello world' and nothing else."):
            event = result.event
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)

        full_response = "".join(chunks)
        self.assertIn("Hello world", full_response)

    def test_stream_with_tool_call(self):
        """Verify streaming execution handles tool calls correctly."""

        # 1. Define Tool
        def get_topic_info(topic: str) -> str:
            return f"Information about {topic}"

        tool = Tool(
            name="get_info",
            description="Get information about a topic",
            args_schema=[
                ArgsSchema(name="topic", type=str, description="The topic to explain")
            ],
            func=get_topic_info,
        )

        # 2. Setup Agent
        agent = Agent(
            llm=self.llm,
            system_prompt="You are a helpful assistant. Use get_info tool when asked.",
            tools=[tool],
        )

        # 3. Stream Execution
        tool_call_finished = False
        final_text = ""

        for result in agent.stream(user_input="Get information about Python"):
            event = result.event

            # Track key events to verify flow
            if event.type == "response.function_call_arguments.done":
                tool_call_finished = True
            elif event.type == "response.output_text.delta":
                final_text += event.delta

        # 4. Assert
        self.assertTrue(tool_call_finished, "Tool call arguments should be completed.")
        self.assertIn("Python", final_text)

    def test_empty_input_error(self):
        """Verify that empty user input raises a ValueError."""
        agent = Agent(llm=self.llm, system_prompt="Booting...")

        with self.assertRaises(ValueError):
            # Consuming generator to trigger execution
            list(agent.stream(user_input=""))


if __name__ == "__main__":
    unittest.main()
