import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import Agent, Tool, Arg, StreamEventType, EventPhase

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Mock function to get weather.
    """
    return f"The weather in {location} is 25 degrees {unit}."

def main():
    # 1. Define a tool
    weather_tool = Tool(
        name="get_weather",
        description="Get the weather for a location",
        func=get_weather,
        args=[
            Arg("location", str, "The city and state, e.g. San Francisco, CA"),
            Arg("unit", str, "The unit of temperature", enum=["celsius", "fahrenheit"]),
        ]
    )

    # 2. Initialize agent
    agent = Agent(
        model="gpt-4o",
        system_prompt="You are a helpful assistant that can check the weather.",
        temperature=0.7
    )
    
    # 3. Register the tool
    agent.add_tool(weather_tool)

    # 4. Run agent with streaming
    print("User: What is the weather in Tokyo? Tell me a short story about it.")
    print("Agent: ", end="", flush=True)
    try:
        for event in agent.stream("What is the weather in Tokyo? Tell me a short story about it."):
            # Handle text deltas
            if event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
                print(event.text, end="", flush=True)
            
            # Handle tool calls (optional logging)
            elif event.type == StreamEventType.TOOL_CALL and event.phase == EventPhase.FINAL:
                print(f"\n[Tool Call: {event.tool_name}({event.arguments})]", flush=True)
            
            # Handle tool outputs (not explicitly in stream events unless we add them, 
            # but the agent loop handles execution internally. 
            # The stream will continue with the tool output processing)
            
        print()
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
