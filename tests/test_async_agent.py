import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, Tool, ChatOpenAI, ArgsSchema
from literun.results import RunResult

# Check if API key is set
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None


@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestAsyncAgentIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for Async Agent execution using the real OpenAI API.
    WARNING: These tests incur costs.
    """

    async def asyncSetUp(self):
        """Initialize common checking tools for tests."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    async def asyncTearDown(self):
        """Clean up resources."""
        if hasattr(self, "llm"):
            await self.llm.aclose()

    async def test_ainvoke_basic(self):
        """Test basic async invocation without tools."""
        agent = Agent(llm=self.llm, system_prompt="You are a helpful assistant.")

        # Real API call
        result = await agent.ainvoke(user_input="Say 'Hello Async'")

        self.assertIsInstance(result, RunResult)
        self.assertIn("Hello Async", result.final_output)

    async def test_ainvoke_with_tool(self):
        """Test async invocation with tool execution."""

        # 1. Define Tool
        def echo(msg: str) -> str:
            return f"Echo: {msg}"

        tool = Tool(
            name="echo",
            description="Echoes the message back",
            func=echo,
            args_schema=[ArgsSchema(name="msg", type=str, description="Message")],
        )

        agent = Agent(llm=self.llm, tools=[tool], system_prompt="Use the echo tool.")

        # Real API call
        result = await agent.ainvoke(user_input="Echo 'Async Works'")

        self.assertIn("Async Works", result.final_output)

    async def test_astream_basic(self):
        """Test basic async streaming."""
        agent = Agent(llm=self.llm, system_prompt="You are a helpful assistant.")

        chunks = []
        async for result in agent.astream(user_input="Say 'Streaming'"):
            event = result.event
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)

        full_text = "".join(chunks)
        self.assertIn("Streaming", full_text)

    async def test_astream_with_tool(self):
        """Test async streaming with tool execution."""

        def get_weather(loc: str):
            return "Sunny"

        tool = Tool(
            name="get_weather",
            description="Get weather",
            func=get_weather,
            args_schema=[ArgsSchema(name="loc", type=str, description="Location")],
        )

        agent = Agent(llm=self.llm, tools=[tool], system_prompt="Use the tool.")

        tool_called = False
        async for result in agent.astream(user_input="Weather in London?"):
            if result.event.type == "response.function_call_arguments.done":
                tool_called = True

        self.assertTrue(tool_called)


if __name__ == "__main__":
    unittest.main()
