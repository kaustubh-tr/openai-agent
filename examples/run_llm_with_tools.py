import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI, Tool, ArgsSchema


# 1. Define tool functions
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather for a given location and unit.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature, either 'celsius' or 'fahrenheit'
    """
    return f"The weather in {location} is 25 degrees {unit}."


# 2. Create tool definition
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

# 3. Configure the LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0,
    timeout=10,
    max_output_tokens=1000,
)

def main():
    # 4. Bind tools to the LLM
    # This registers the tools with the model instance for all subsequent calls
    llm.bind_tools(tools=[weather_tool])

    user_input = "What is the weather in Tokyo?"
    # user_input = "Tell me a joke?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    
    print("--- LLM with Bound Tools ---")

    # 5. Invoke the LLM
    print(f"User: {user_input}")

    try:
        # The LLM will return a response that may contain tool_calls
        response = llm.invoke(messages=messages)

        # Check if the model decided to call a tool
        for item in response.output:
            if item.type == "function_call":
                print(f"Tool Calls:")
                print(f"  - Name: {item.name}")
                print(f"  - Arguments: {item.arguments}")

            if item.type == "message":
                text_parts = [
                    c.text for c in item.content if c.type == "output_text"
                ]
                final_output_text = "".join(text_parts)
                print(f"Assistant: {final_output_text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
