import unittest
from src.openai_agent import Agent, ChatOpenAI, Tool, ArgsSchema, Response
from src.openai_agent.tool import ToolRuntime
from typing import Any, Dict

class TestRuntimeContext(unittest.TestCase):
    def test_invoke_with_context(self):
        # 1. Define a tool that accepts ToolRuntime
        def get_user_id(ctx: ToolRuntime) -> str:
            # Manually extract from context (attribute access now)
            user_id = ctx.user_id
            return f"User ID is {user_id}"

        tool = Tool(
            name="get_user_id",
            description="Get the current user ID",
            func=get_user_id,
            args_schema=[] # No LLM args
        )
        
        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])
        # agent.add_tool(tool)
        
        # 3. Test execution with runtime_context
        context = {"user_id": 42}
        
        # We test _execute_tool directly
        result = agent._execute_tool("get_user_id", {}, runtime_context=context)
        self.assertEqual(result, "User ID is 42")

    def test_mixed_args_and_context(self):
        # Tool needing both LLM arg and Context
        def multiply_by_user_factor(x: int, ctx: ToolRuntime) -> int:
            factor = getattr(ctx, "factor", 1)
            return x * factor
            
        tool = Tool(
            name="multiply",
            description="Multiplies by user factor",
            func=multiply_by_user_factor,
            args_schema=[ArgsSchema(name="x", type=int, description="Input")]
        )
        
        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])
        
        context = {"factor": 3}
        # LLM gives x=10, Runtime gives factor=3
        result = agent._execute_tool("multiply", {"x": 10}, runtime_context=context)
        self.assertEqual(result, 30)
    
    def test_missing_context_arg(self):
        # If tool expects context but uses it blindly, it might crash if context is empty, 
        # but the Agent should always pass the ToolRuntime object (maybe empty).
        def check_presence(ctx: ToolRuntime) -> str:
            # Check attribute existence
            if hasattr(ctx, "secret"):
                 return "Found"
            return "Not found"
            
        tool = Tool(
            name="check",
            description="Check secret",
            func=check_presence,
            args_schema=[]
        )
        agent = Agent(llm=ChatOpenAI(api_key="fake"), tools=[tool])
        
        # Pass None as context -> agent defaults to {}
        result = agent._execute_tool("check", {}, runtime_context=None)
        self.assertEqual(result, "Not found")

if __name__ == '__main__':
    unittest.main()
