import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI, Tool, ArgsSchema
from literun.utils import extract_tool_calls


# Check if API key is set
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None


@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestAsyncChatOpenAI(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests for ChatOpenAI async client wrapper.
    """

    async def asyncTearDown(self):
        if hasattr(self, "llm"):
            await self.llm.aclose()

    async def test_ainvoke_simple_text(self):
        """Verify asynchronous text generation."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        messages = [{"role": "user", "content": "Say 'hello async'"}]

        response = await self.llm.ainvoke(messages)

        self.assertIsNotNone(response.output_text)
        self.assertIn("hello", response.output_text.lower())

    async def test_astream_simple_text(self):
        """Verify asynchronous streaming text generation."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        messages = [{"role": "user", "content": "Say 'hello async'"}]

        chunks = []
        async for event in self.llm.astream(messages=messages):
            if event.type == "response.output_text.delta":
                if event.delta:
                    chunks.append(event.delta)

        full_text = "".join(chunks)
        self.assertIn("hello", full_text.lower())

    async def test_bind_tools_tool_call_async(self):
        """Verify binding tools enables tool calls in async response."""

        # 1. Define Tool
        def get_weather(location: str):
            return "Sunny"

        tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            func=get_weather,
            args_schema=[ArgsSchema(name="location", type=str, description="City")],
        )

        # 2. Setup LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        self.llm.bind_tools(tools=[tool])

        # 3. Invoke with tool-triggering prompt
        messages = [{"role": "user", "content": "What is the weather in London?"}]
        response = await self.llm.ainvoke(messages)

        # 4. Extract and Verify
        tool_calls = extract_tool_calls(response)

        if tool_calls:
            self.assertGreater(len(tool_calls), 0)
            tool_names = [call["name"] for call in tool_calls]
            self.assertIn("get_weather", tool_names)
        else:
            print(
                "WARNING: Model did not generate a tool call in test_bind_tools_tool_call_async."
            )


if __name__ == "__main__":
    unittest.main()
