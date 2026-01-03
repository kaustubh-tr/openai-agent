# OpenAI Agent Framework

A lightweight, flexible Python framework for building custom OpenAI agents (Responses API) with tool support and structured prompt management.

## Features

- **Custom Agent Loop**: Full control over the chat-completion loop.
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

### Basic Example

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

See [examples/basic_agent.py](examples/basic_agent.py) for a complete runnable example.

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
