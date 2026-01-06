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
from openai_agent import Agent, Tool, Arg

# 1. Define a function
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

# 2. Wrap it in a Tool
weather_tool = Tool(
    name="get_weather",
    description="Get weather info",
    func=get_weather,
    args=[Arg("location", str, "City name")]
)

# 3. Initialize Agent
agent = Agent(
    model="gpt-4o",
    system_prompt="You are a helpful assistant."
)
agent.add_tool(weather_tool)

# 4. Run
response = agent.invoke("What's the weather in Paris?")
print(response)
```

### Streaming Example

The agent supports streaming responses with event filtering.

```python
from openai_agent import Agent, Tool, Arg, StreamEventType, EventPhase

# ... (Tool definition same as above) ...

agent = Agent(
    model="gpt-4o",
    system_prompt="You are a helpful assistant."
)
agent.add_tool(weather_tool)

print("Agent: ", end="", flush=True)

# Stream returns a generator of StreamEvent objects
# Set include_internal_events=True to receive raw OpenAI API events in the `raw_event` attribute of each StreamEvent
for event in agent.stream("What's the weather in Tokyo?", include_internal_events=False):
    
    # Handle text content updates
    if event.type == StreamEventType.TEXT and event.phase == EventPhase.DELTA:
        print(event.text, end="", flush=True)
        
    # Handle tool usage (optional)
    elif event.type == StreamEventType.TOOL_CALL and event.phase == EventPhase.FINAL:
        print(f"\n[Tool used: {event.tool_name}]", end="\nAgent: ", flush=True)
        
print()
```

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
