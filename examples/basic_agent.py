import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import Agent, Tool, Arg

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Mock function to get weather.
    In a real app, this would call an external API.
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
    # Make sure you have OPENAI_API_KEY set in your environment variables
    agent = Agent(
        model="gpt-4o",
        system_prompt="You are a helpful assistant that can check the weather.",
        temperature=0.7
    )
    
    # 3. Register the tool
    agent.add_tool(weather_tool)

    # 4. Run agent
    print("User: What is the weather in Tokyo?")
    try:
        response = agent.invoke("What is the weather in Tokyo?")
        print(f"Agent: {response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
