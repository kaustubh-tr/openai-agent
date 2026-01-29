# Literun 🚀

A lightweight, flexible Python framework for building custom OpenAI agents (Responses API) with tool support and structured prompt management.

## Features

- **Custom Agent Execution**: Complete control over the agent execution loop, supporting both synchronous and streaming responses.
- **Tool Support**: Easy registration and execution of Python functions as tools.
- **Type Safety**: Strong typing for tool arguments with automatic coercion and validation.
- **Prompt Templates**: Structured way to build system, user, and assistant messages.
- **Constants**: Pre-defined constants for OpenAI roles and message types.
- **Streaming Support**: Built-in support for real-time streaming of agent thoughts, tool calls, and responses.
- **Tool Management**: Easy-to-define tools with automatic JSON schema generation (`ArgsSchema`).
- **Event-Driven**: Structured event system for granular control over the agent's execution lifecycle.
- **OpenAI Compatible**: Seamlessly integrates with `openai-python` client.

## Requirements

- Python 3.10+  
- [OpenAI Python API library](https://pypi.org/project/openai/)

## Installation

### Production

```bash
pip install literun
```

### Development

```bash
git clone https://github.com/kaustubh-tr/literun.git
cd openai-agent
pip install -e .[dev]
```

## Quick Start

### Basic Agent

Here is a simple example of how to create an agent with a custom tool:

```python
import os
from literun import Agent, ChatOpenAI, Tool, ArgsSchema

# 1. Define a tool function
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"The weather in {location} is 25 degrees {unit}."

# 2. Wrap it with Tool schema
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

# 3. Initialize LLM and Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 4. Initialize Agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=[weather_tool],
)

# 5. Run the Agent
result = agent.invoke(user_input="What is the weather in Tokyo?")
print(f"Final Answer: {result.final_output}")
```

### Streaming Agent

You can also stream the agent's execution to handle events in real-time:

```python
# ... (setup tool and agent as above)

print("Agent: ", end="", flush=True)
for result in agent.stream(user_input="What is the weather in Tokyo?"):
    event = result.event
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.function_call_arguments.done":
        print(f"\n[Tool Call: {event.name}]")

print()
```

### Runtime Configuration (Context Injection)

The framework allows passing a runtime context to tools using explicit context injection.

Rules:
1. Define a tool function with a parameter annotated with `ToolRuntime`.
2. The framework will automatically inject the `runtime_context` (wrapped in `ToolRuntime`) into that parameter.
3. Access configuration values using `ctx.{parameter}`.

```python
from typing import Dict, Any
from literun import Tool, ArgsSchema, ToolRuntime

# 1. Define tool with context
def get_weather(location: str, ctx: ToolRuntime) -> str:
    """
    Returns weather info for a location.
    The runtime context can include sensitive info like user_id or API keys.
    """
    user_id = getattr(ctx, "user_id", "unknown_user")
    api_key = getattr(ctx, "weather_api_key", None)

    # Simulate fetching weather
    return f"Weather for {location} fetched using API key '{api_key}' for user '{user_id}'."

# 2. Register tool 
tool = Tool(
    name="get_weather",
    description="Get the weather for a given location",
    func=get_weather,
    args_schema=[
        ArgsSchema(
            name="location",
            type=str,
            description="Location for which to get the weather",
        )
    ]
)

# 3. Setup agent
agent = Agent(
    llm=ChatOpenAI(api_key="fake"), 
    tools=[tool]
)

# 4. Pass config at runtime
# The whole dict is passed into the 'ctx' argument
agent.invoke(
    user_input="What's the weather in London?",
    runtime_context={
        "user_id": "user_123",
        "weather_api_key": "SECRET_API_KEY_456"
    }
)
```

### Using ChatOpenAI Directly

You can also use the `ChatOpenAI` class directly if you don't need the agent loop (e.g., for simple, one-off LLM calls).

```python
from literun import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
]

# Synchronous call
# Returns the raw OpenAI Responses API response object
response = llm.invoke(messages=messages)
print(response.output_text)

# Or streaming call
# Returns a generator of raw OpenAI response stream events
stream = llm.stream(messages=messages)
for event in stream:
    print(event)
```

See [examples](examples/) for complete runnable examples.

## Project Structure

The project is organized as follows:

```
literun/
├── src/
│   └── literun/          # Main package source
│       ├── agent.py      # Agent runtime logic
│       ├── llm.py        # LLM client wrapper
│       ├── tool.py       # Tool definition and execution
│       ├── events.py     # Stream event types
│       └── ...
├── tests/                # Unit tests
├── examples/             # Usage examples
└── pyproject.toml        # Project configuration
```

## Testing

Run the test suite using `unittest`:

```bash
python -m unittest discover tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m unittest discover tests`
5. Update the example usage if needed
6. Submit a pull request

## License

MIT
