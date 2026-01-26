import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import Agent, ChatOpenAI, Tool, ArgsSchema

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Mock function to get weather.
    In a real app, this would call an external API.
    """
    return f"The weather in {location} is 25 degrees {unit}."

def main():
    # 1. Define a tool
    weather_tool = Tool(
        func=get_weather,
        name="get_weather",
        description="Get the weather for a location",
        args_schema=[
            ArgsSchema(
                name="location", 
                type=str, 
                description="The city and state, e.g. San Francisco, CA"
            ),
            ArgsSchema(
                name="unit", 
                type=str, 
                description="The unit of temperature", 
                enum=["celsius", "fahrenheit"]
            ),
        ]
    )

    # 2. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # 3. Initialize agent
    # Make sure you have OPENAI_API_KEY set in your environment variables
    agent = Agent(
        llm=llm,
        system_prompt="You are a helpful assistant that can check the weather.",
        tools=[weather_tool]
    )
    
    # 5. Run agent
    print("User: What is the weather in Tokyo?")
    try:
        result = agent.invoke(user_input="What is the weather in Tokyo?")
        print(f"Agent: {result.output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
