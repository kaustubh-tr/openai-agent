# LiteRun 🚀

[![PyPI - Version](https://img.shields.io/pypi/v/literun)](https://pypi.org/project/literun/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/literun)](https://pypi.org/project/literun/)
[![PyPI - License](https://img.shields.io/pypi/l/literun)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-DOCS-blue)](https://github.com/kaustubh-tr/literun/blob/main/DOCS.md)

A lightweight, flexible Python framework for building custom OpenAI agents (Responses API) with tool support and structured prompt management.

## Features

- **Custom Agent Execution**: Control the loop with synchronous and streaming support.
- **Tool Support**: Easy registration with Pydantic-powered validation.
- **Type Safety**: Built for modern Python 3.10+ environments.
- **Prompt Templates**: Structured message management.
- **Event-Driven**: Granular control via a rich event system.

For detailed documentation on Architecture, Streaming, and Advanced Configuration, see [DOCS.md](https://github.com/kaustubh-tr/literun/blob/main/DOCS.md).

## Requirements

- Python 3.10+

> **Note**: Core dependencies like `openai` and `pydantic` are automatically installed when you install `literun`.

## Installation

You can install `literun` directly from PyPI:

```bash
pip install literun
```

## Quick Start

### Basic Agent

Here is a simple example of how to create an agent with a custom tool.

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

# 3. Initialize Agent
agent = Agent(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.7),
    system_prompt="You are a helpful assistant.",
    tools=[weather_tool],
)

# 4. Run the Agent
result = agent.invoke(user_input="What is the weather in Tokyo?")
print(f"Final Answer: {result.final_output}")
```

### Advanced Usage

LiteRun supports **Streaming**, **Runtime Context Injection** (for secrets), and **Direct LLM Usage**.

👉 Check out the [Documentation](https://github.com/kaustubh-tr/literun/blob/main/DOCS.md) and [Examples](https://github.com/kaustubh-tr/literun/blob/main/examples/) for more details.

## Project Structure

```text
literun/
├── src/
│   └── literun/          # Main package source
│       ├── agent.py      # Agent orchestrator
│       ├── llm.py        # ChatOpenAI wrapper
│       ├── tool.py       # Tool & Schema definitions
│       └── ...
├── tests/                # Unit tests (agent, llm, tools, prompts)
├── examples/             # Runnable examples
├── DOCS.md               # Detailed documentation
├── LICENSE               # MIT License
├── README.md             # This file
└── pyproject.toml        # Project configuration & dependencies
```

## Contributing

We welcome contributions! Please follow these steps to set up your development environment:

1.  **Fork** the repository and clone it locally:

    ```bash
    git clone https://github.com/kaustubh-tr/literun.git
    cd literun
    ```

2.  **Install** in editable mode with development dependencies:

    ```bash
    pip install -e .[dev]
    ```

3.  **Create** a feature branch and make your changes.

4.  **Test** your changes (see below).

5.  **Submit** a pull request.

## Testing

This project uses `pytest` as the primary test runner, but supports `unittest` as well.

```bash
# Run all tests
pytest
```

or using unittest:

```bash
python -m unittest discover tests
```

> **Note**: Some integration tests require the `OPENAI_API_KEY` environment variable. They are automatically skipped if it is missing.

## License

Copyright (c) 2026 Kaustubh Trivedi.

Distributed under the terms of the [MIT](https://github.com/kaustubh-tr/literun/blob/main/LICENSE) license, LiteRun is free and open source software.
