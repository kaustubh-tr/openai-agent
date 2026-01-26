# OpenAI Agent Framework

A lightweight, flexible Python framework for building custom OpenAI agents (Responses API) with tool support and structured prompt management.

## Features

- **Custom Agent Execution**: Complete control over the agent execution loop, supporting both synchronous and streaming responses.
- **Tool Support**: Easy registration and execution of Python functions as tools.
- **Type Safety**: Strong typing for tool arguments with automatic coercion and validation.
- **Prompt Templates**: Structured way to build system, user, and assistant messages.
- **Constants**: Pre-defined constants for OpenAI roles and message types.

## Requirements

- Python 3.10+  
- [OpenAI Python API library](https://pypi.org/project/openai/)

## Installation

### Production

```bash
pip install openai-agent
```

### Development

```bash
git clone https://github.com/kaustubh-tr/openai-agent.git
cd openai-agent
pip install -e .[dev]
```

## Usage

### Basic Example (Synchronous)

```python
from openai_agent import Agent, ChatOpenAI, Tool, ArgsSchema

# 1. Define a function
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

# 2. Wrap it in a Tool
weather_tool = Tool(
    name="get_weather",
    description="Get weather info",
    func=get_weather,
    args_schema=[ArgsSchema(name="location", type=str, description="City name")]
)

# 3. Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 4. Initialize Agent
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=[weather_tool]
)

# 5. Run
# Returns a Response object
response = agent.invoke(user_input="What's the weather in Paris?")
print(response.output)
```

### Streaming Example

The agent supports streaming responses with event filtering.

```python
from openai_agent import Agent, ChatOpenAI, Tool, ArgsSchema, StreamEventType, EventPhase

# ... (Tool definition same as above) ...
weather_tool = Tool(
    name="get_weather",
    description="Get weather info",
    func=get_weather,
    args_schema=[ArgsSchema(name="location", type=str, description="City name")]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=[weather_tool]
)

print("Agent: ", end="", flush=True)

# Stream returns a generator of ResponseStreamEvent objects
# Set include_internal_events=True to receive all OpenAI API events in the `raw_event` attribute of each ResponseStreamEvent
for event in agent.stream(user_input="What's the weather in Tokyo?", include_internal_events=False):
    
    # Handle text content updates
    if event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
        print(event.text, end="", flush=True)
        
    # Handle tool usage (optional)
    elif event.type == StreamEventType.TOOL_CALL and event.phase == EventPhase.FINAL:
        print(f"\n[Tool used: {event.tool_name}]", end="\nAgent: ", flush=True)
        
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
from openai_agent import Tool, ArgsSchema, ToolRuntime

# 1. Define tool with context
def get_user_data(user_id: str, ctx: ToolRuntime) -> str:
    db_conn = getattr(ctx, "db_conn", None)
    return f"Fetching data for {user_id} using {db_conn}..."

# 2. Register tool 
# Only expose 'user_id' to the LLM
tool = Tool(
    name="get_user_data",
    description="Get user info",
    func=get_user_data,
    args_schema=[ArgsSchema("user_id", str, "ID of the user")]
)

agent = Agent(
    llm=ChatOpenAI(api_key="fake"), 
    tools=[tool]
)

# 3. Pass config at runtime
# The whole dict is passed into the 'ctx' argument
agent.invoke(
    user_input="Who is user 123?",
    runtime_context={"db_conn": "ProductionDB"}
)
```

### Using ChatOpenAI Directly

You can also use the `ChatOpenAI` class directly if you don't need the agent loop (e.g., for simple, one-off LLM calls).

```python
from openai_agent import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."}
]

# Synchronous call
# Returns a Response object containing output, usage, tool_calls etc.
response = llm.invoke(messages)
print(response.output)

# Or streaming call
# Returns generator of ResponseStreamEvent
stream = llm.stream(messages=messages)
for event in stream:
    # ... handle events ...
    pass
```

## Key Classes

### Response
The unified output object for synchronous calls (`agent.invoke`, `llm.invoke`).
- `output` (str): The text response.
- `tool_calls` (List[Dict]): List of tool calls made.
- `usage` (Dict): Token usage statistics.

### ResponseStreamEvent
The unified event object for streaming calls (`agent.stream`, `llm.stream`).
- `type` (StreamEventType): The type of event (TEXT, TOOL_CALL, LIFECYCLE, ERROR).
- `phase` (EventPhase): The phase of the event (DELTA, FINAL).
- `text` (str): Text content (for text events).
- `usage` (Dict): Usage stats (on lifecycle events).

See [examples/](examples/) for complete runnable examples.

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
5. Submit a pull request

## License

MIT
