
import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import Agent, Tool, Arg

# Check if API key is set
SKIP_REAL_API_TESTS = os.getenv("OPENAI_API_KEY") is None

@unittest.skipIf(SKIP_REAL_API_TESTS, "OPENAI_API_KEY not set")
class TestAgent(unittest.TestCase):
    def test_initialization(self):
        agent = Agent(model="gpt-4o", system_prompt="Test system prompt")
        self.assertEqual(agent.model, "gpt-4o")
        self.assertEqual(agent.system_prompt, "Test system prompt")
        self.assertEqual(agent.tools, {})

    def test_add_tool(self):
        agent = Agent(model="gpt-4o")
        tool = Tool("test_tool", "desc", [], lambda: None)
        agent.add_tool(tool)
        self.assertIn("test_tool", agent.tools)
        
        # Test duplicate registration
        with self.assertRaises(ValueError):
            agent.add_tool(tool)

    def test_invoke_simple_response(self):
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
        response = agent.invoke("Say 'Hello world' and nothing else.")
        self.assertIsInstance(response, str)
        self.assertIn("Hello world", response)

    def test_invoke_with_tool_call(self):
        # Setup tool
        def echo(msg: str) -> str:
            return f"Echo: {msg}"
        
        tool = Tool(
            name="echo", 
            description="Echoes the message back to the user", 
            args=[Arg("msg", str, "The message to echo")], 
            func=echo
        )
        
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant. Use the echo tool when asked.")
        agent.add_tool(tool)

        response = agent.invoke("Please use the echo tool to say 'hello'")
        self.assertIn("Echo: hello", response)

    def test_stream_simple_response(self):
        from src.openai_agent.constants import StreamEventType, EventPhase
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
        chunks = []
        for event in agent.stream("Say 'Hello world' and nothing else."):
            if event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
                chunks.append(event.text)
        response = "".join(chunks)
        self.assertIn("Hello world", response)

if __name__ == "__main__":
    unittest.main()
