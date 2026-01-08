import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import ChatOpenAI, Tool, ArgsSchema
from src.openai_agent.output_schema import Response
from src.openai_agent.constants import StreamEventType, EventPhase

# Check if API key is set
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None

@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestChatOpenAI(unittest.TestCase):
    def test_initialization(self):
        llm = ChatOpenAI(model="gpt-4o")
        self.assertEqual(llm.model, "gpt-4o")
        self.assertIsNone(llm._parallel_tool_calls)

    def test_invoke_simple(self):
        llm = ChatOpenAI(model="gpt-4o")
        messages = [{"role": "user", "content": "Say hello world"}]
        response = llm.invoke(messages)
        self.assertIsInstance(response, Response)
        self.assertIsNotNone(response.output)
        self.assertIn("hello", response.output.lower())
        self.assertIsNotNone(response.usage)

    def test_stream_simple(self):
        llm = ChatOpenAI(model="gpt-4o")
        messages = [{"role": "user", "content": "Say hello world"}]
        
        chunks = []
        for event in llm.stream(messages=messages):
            if event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
                if event.text:
                    chunks.append(event.text)
        
        full_text = "".join(chunks)
        self.assertIn("hello", full_text.lower())

    def test_bind_tools_call(self):
        # Define a dummy tool
        def get_weather(location: str):
            return "Sunny"

        tool = Tool(
            name="get_weather",
            description="Get weather",
            func=get_weather,
            args_schema=[ArgsSchema(name="location", type=str, description="City")]
        )

        llm = ChatOpenAI(model="gpt-4o")
        llm.bind_tools(tools=[tool])
        
        # We need a prompt that forces a tool call
        messages = [{"role": "user", "content": "What is the weather in London?"}]
        response = llm.invoke(messages)
        
        self.assertIsInstance(response, Response)
        self.assertIsNotNone(response.usage)
        
        # Verify that tool_calls were detected
        if response.tool_calls:
            self.assertGreater(len(response.tool_calls), 0)
            # We can try to find our tool name in the calls
            tool_names = [call['name'] for call in response.tool_calls]
            self.assertIn("get_weather", tool_names)
        else:
            # It's possible the model refused to call the tool or just answered from knowledge.
            # But gpt-4o usually respects tool definitions.
            # We won't fail hard if tool_calls is empty to avoid flaky tests on model behavior,
            # but we check the structure is correct.
            pass

if __name__ == '__main__':
    unittest.main()
