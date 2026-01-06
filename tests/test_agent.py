
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

    def test_stream_with_tool_call(self):
        """Test streaming with tool calls"""
        from src.openai_agent.constants import StreamEventType, EventPhase, ProcessStatus
        
        # Setup tool
        def get_info(topic: str) -> str:
            return f"Information about {topic}"
        
        tool = Tool(
            name="get_info", 
            description="Get information about a topic", 
            args=[Arg("topic", str, "The topic to get information about")], 
            func=get_info
        )
        
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant. Use get_info tool when asked for information.")
        agent.add_tool(tool)
        
        # Track events
        tool_call_events = []
        lifecycle_events = []
        text_chunks = []
        
        for event in agent.stream("Get information about Python and tell me about it"):
            if event.type == StreamEventType.TOOL_CALL:
                tool_call_events.append(event)
            elif event.type == StreamEventType.LIFECYCLE:
                lifecycle_events.append(event)
            elif event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
                text_chunks.append(event.text)
        
        # Verify tool calls were made
        self.assertGreater(len(tool_call_events), 0, "Should have at least one tool call event")
        
        # Verify lifecycle events
        self.assertGreater(len(lifecycle_events), 0, "Should have lifecycle events")
        started_events = [e for e in lifecycle_events if e.process_status == ProcessStatus.STARTED]
        completed_events = [e for e in lifecycle_events if e.process_status == ProcessStatus.COMPLETED]
        self.assertGreater(len(started_events), 0, "Should have at least one started lifecycle event")
        self.assertGreater(len(completed_events), 0, "Should have at least one completed lifecycle event")
        
        # Verify tool call phases
        tool_call_phases = [e.phase for e in tool_call_events]
        self.assertIn(EventPhase.FINAL, tool_call_phases, "Should have final tool call phase")
        
        # Verify final response contains tool output
        response = "".join(text_chunks)
        self.assertIn("Python", response)

    def test_stream_lifecycle_events(self):
        """Test that lifecycle events are emitted correctly"""
        from src.openai_agent.constants import StreamEventType, ProcessStatus
        
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
        
        lifecycle_events = []
        for event in agent.stream("Say hello"):
            if event.type == StreamEventType.LIFECYCLE:
                lifecycle_events.append(event)
        
        # Should have started and completed events
        self.assertGreater(len(lifecycle_events), 0, "Should have lifecycle events")
        
        statuses = [e.process_status for e in lifecycle_events]
        self.assertIn(ProcessStatus.STARTED, statuses, "Should have STARTED status")
        self.assertIn(ProcessStatus.COMPLETED, statuses, "Should have COMPLETED status")
        
        # Verify response_id is present
        for event in lifecycle_events:
            self.assertIsNotNone(event.response_id, "Lifecycle events should have response_id")

    def test_stream_internal_events(self):
        """Test that internal events are emitted when include_internal_events=True"""
        from src.openai_agent.constants import StreamEventType
        
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
        
        internal_events = []
        for event in agent.stream("Say hi", include_internal_events=True):
            if event.type == StreamEventType.INTERNAL:
                internal_events.append(event)
        
        # Should have internal events when flag is True
        self.assertGreater(len(internal_events), 0, "Should have internal events when include_internal_events=True")
        
        # Verify raw_event is present
        for event in internal_events:
            self.assertIsNotNone(event.raw_event, "Internal events should have raw_event")

    def test_stream_empty_input(self):
        """Test that streaming with empty input raises ValueError"""
        agent = Agent(model="gpt-4o", system_prompt="You are a helpful assistant.")
        
        with self.assertRaises(ValueError) as context:
            list(agent.stream(""))
        
        self.assertIn("cannot be empty", str(context.exception))

if __name__ == "__main__":
    unittest.main()
