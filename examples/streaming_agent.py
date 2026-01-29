import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, ChatOpenAI, Tool, ArgsSchema


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Mock function to get weather.
    """
    return f"The weather in {location} is 25 degrees {unit}."


def main():
    # 1. Define a tool with args schema
    weather_tool = Tool(
        func=get_weather,
        name="get_weather",
        description="Get the weather for a location",
        args_schema=[
            ArgsSchema(
                name="location",
                type=str,
                description="The city and state, e.g. San Francisco, CA",
            ),
            ArgsSchema(
                name="unit",
                type=str,
                description="The unit of temperature",
                enum=["celsius", "fahrenheit"],
            ),
        ],
    )

    # 2. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # 3. Initialize agent
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant that can check the weather.",
        tools=[weather_tool],
    )

    # 4. Run agent with streaming
    print("User: What is the weather in Tokyo? Tell me a short story about it.")
    print("Agent: ", end="", flush=True)
    try:
        for result in agent.stream(
            user_input="What is the weather in Tokyo? Tell me a short story about it."
        ):
            event = result.event
            # print(event)  # for complete list of events for this agent run
            if getattr(event, "type", None) == "response.output_text.delta":
                delta_text = getattr(event, "delta", "")
                if isinstance(delta_text, str):
                    print(delta_text, end="", flush=True)

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
