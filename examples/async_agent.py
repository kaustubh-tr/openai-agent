import asyncio
import os
import sys

# Ensure the package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from literun import Agent, ChatOpenAI, Tool, ArgsSchema


# 1. Define an async tool
async def get_weather(location: str) -> str:
    """Async tool to simulate API call."""
    print(f"  [Async Tool] Fetching weather for {location}...")
    await asyncio.sleep(1)  # Simulate network delay
    return f"The weather in {location} is 25°C and Sunny."


weather_tool = Tool(
    name="get_weather",
    description="Get the weather for a location",
    args_schema=[ArgsSchema(name="location", type=str)],
    coroutine=get_weather,
)


# 2. Define a sync tool (will run in thread)
def calculator(a: int, b: int) -> int:
    """Sync tool execution."""
    print(f"  [Sync Tool] Calculating {a} + {b}...")
    return a + b


calc_tool = Tool(
    name="calculator",
    description="Add two numbers",
    args_schema=[ArgsSchema(name="a", type=int), ArgsSchema(name="b", type=int)],
    func=calculator,
)


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
        return
    
    # 3. Configure your model
    llm = ChatOpenAI(
        model="gpt-4.1-mini", 
        temperature=0.7,
        timeout=10,
        max_output_tokens=1000,
    )
    
    # 4. Create the agent
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant.",
        tools=[weather_tool, calc_tool],
        max_iterations=5,
    )
    
    # 5. Run agent with async tool and sync tool
    print("\n--- Run agent ainvoke (Async Tool) ---")
    result = await agent.ainvoke(user_input="What is the weather in Mumbai?")
    print("Agent Response:", result.final_output)

    print("\n--- Run agent ainvoke (Sync Tool) ---")
    result = await agent.ainvoke(user_input="Calculate 50 + 100")
    print("Agent Response:", result.final_output)

    # 6. Run agent with async streaming
    print("\n--- Run agent astream ---")
    print("Agent Response: ", end="", flush=True)
    async for event in agent.astream(
        user_input="Tell me a very short joke about async programming."
    ):
        if event.event.type == "response.output_text.delta":
            print(event.event.delta, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
