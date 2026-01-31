import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import Agent, ChatOpenAI, Tool, ArgsSchema, ToolRuntime, PromptTemplate


# 1. Define the system prompt
SYSTEM_PROMPT = """You are a helpful assistant.
You have access to the following tools:
- `search_database` to search the customer database.
- `get_weather` to get the weather for a location.
Use the tools as needed to answer user queries.
Be concise and informative in your responses.
"""


# 2. Create tool functions
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather for a given location and unit.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The unit of temperature, either 'celsius' or 'fahrenheit'
    """
    return f"The weather in {location} is 25 degrees {unit}."


def search_database(query: str, limit: int, runtime: ToolRuntime) -> str:
    """Search the customer database for records matching the query.
    
    This tool demonstrates Runtime Context'. The `request_id` and `user_id` are not provided
    by the LLM, but injected by the application at runtime.
    
    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    user_id = runtime.user_id
    request_id = runtime.request_id
    return f"Found {limit} results for '{query}' for User {user_id} [Req: {request_id}]"


# 3. Create tools
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

search_database_tool = Tool(
    func=search_database,
    name="search_database",
    description="Search the customer database for records matching the query",
    args_schema=[
        ArgsSchema(
            name="query",
            type=str,
            description="Search terms to look for",
        ),
        ArgsSchema(
            name="limit",
            type=int,
            description="Maximum number of results to return",
        ),
    ]
)

# 4. Configure your model
llm = ChatOpenAI(
    model="gpt-4.1-mini", 
    temperature=0.7,
    timeout=10,
    max_output_tokens=1000,
)

# 5. Create the agent
agent = Agent(
    llm=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[weather_tool, search_database_tool],
    max_iterations=5,
)

def main():
    # 6. Define the prompt template with chat history
    user_input = "Find customer records for John Doe with user id and request id."
    
    prompt = PromptTemplate()
    prompt.add_user("What is the weather in Tokyo? Tell me a short story about it.")
    prompt.add_tool_call(
        name="get_weather",
        arguments=str({"location": "Tokyo", "unit": "celsius"}),
        call_id="call_123",
    )
    prompt.add_tool_output(call_id="call_123", output="The weather in Tokyo is 25 degrees celsius.")
    prompt.add_assistant(
        "The weather in Tokyo is currently 25 degrees Celsius, a warm and pleasant day.\n\n"
        "Short story:\n"
        "On a bright and breezy day in Tokyo, the city buzzed with life under the gentle warmth of "
        "25 degrees Celsius. Sakura blossoms danced lightly in the soft wind as people strolled "
        "through parks, savoring the perfect spring weather. A young artist, inspired by the clear "
        "skies and vibrant atmosphere, set up her easel by the river, capturing the essence of the "
        "lively city in her painting. It was a day where nature and urban life blended beautifully, "
        "creating a moment to remember."
    )
    
    print("\n\n--- Non-Streaming Agent Response ---\n\n")
    
    # 7. Run agent without streaming
    print(f"User: {user_input}")
    try:
        result = agent.invoke(
            user_input=user_input,
            prompt_template=prompt,
            runtime_context={
                "user_id": "user-123", 
                "request_id": "req-456"
            },
        )
        # print(f"Agent: \n{result.new_items}")  # for complete list of items for this agent run
        print(f"Agent: {result.final_output}")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n\n--- Streaming Agent Response ---\n\n")
    
    # 8. Run agent with streaming
    print(f"User: {user_input}")
    print("Agent: ", end="", flush=True)
    try:
        for result in agent.stream(
            user_input=user_input,
            prompt_template=prompt,
            runtime_context={
                "user_id": "user-123", 
                "request_id": "req-456"
            },
        ):
            event = result.event
            # print(event)  # for complete list of events for this agent run
            if getattr(event, "type", None) == "response.output_text.delta":
                delta_text = getattr(event, "delta", "")
                if isinstance(delta_text, str):
                    print(delta_text, end="", flush=True)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
