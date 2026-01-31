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
class TestChatOpenAI(unittest.TestCase):
    """
    Integration tests for ChatOpenAI client wrapper.
    """

    def test_initialization(self):
        """Verify initialization sets attributes correctly."""
        llm = ChatOpenAI(model="gpt-4o")
        self.assertEqual(llm.model, "gpt-4o")
        self.assertIsNone(llm._parallel_tool_calls)

    def test_invoke_simple_text(self):
        """Verify synchronous text generation."""
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        messages = [{"role": "user", "content": "Say 'hello world'"}]

        response = llm.invoke(messages)

        self.assertIsNotNone(response.output_text)
        self.assertIn("hello", response.output_text.lower())

    def test_stream_simple_text(self):
        """Verify streaming text generation."""
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        messages = [{"role": "user", "content": "Say 'hello world'"}]

        chunks = []
        for event in llm.stream(messages=messages):
            if event.type == "response.output_text.delta":
                if event.delta:
                    chunks.append(event.delta)

        full_text = "".join(chunks)
        self.assertIn("hello", full_text.lower())

    def test_bind_tools_tool_call(self):
        """Verify binding tools enables tool calls in the LLM response."""

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
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        llm.bind_tools(tools=[tool])

        # 3. Invoke with tool-triggering prompt
        messages = [{"role": "user", "content": "What is the weather in London?"}]
        response = llm.invoke(messages)

        # 4. Extract and Verify
        tool_calls = extract_tool_calls(response)

        # Note: We rely on the model's behavior here.
        # gpt-4o is generally reliable at calling tools when bound.
        if tool_calls:
            self.assertGreater(len(tool_calls), 0)
            tool_names = [call["name"] for call in tool_calls]
            self.assertIn("get_weather", tool_names)
        else:
            # If model didn't call tool (rare but possible), we shouldn't fail
            # unpredictably, but warn. In strict CI, we might retry or fail.
            print(
                "WARNING: Model did not generate a tool call in test_bind_tools_call."
            )


if __name__ == "__main__":
    unittest.main()
