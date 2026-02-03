# LiteRun Documentation 📚

LiteRun is a lightweight, flexible Python framework for building custom OpenAI agents. It provides a robust abstraction over the OpenAI Chat Completions API, adding tool management, structured prompt handling, and event-driven execution without the bloat of larger frameworks.

## Table of Contents

- [Core Architecture](#core-architecture)
- [Execution Modes](#execution-modes)
- [Agent Reference](#agent-reference)
- [Tool Management](#tool-management)
- [ChatOpenAI Reference](#chatopenai-reference)
- [Runtime Context Injection](#runtime-context-injection)
- [Prompt Templates](#prompt-templates)
- [Streaming](#streaming)

---

## Core Architecture

LiteRun is built around three main components:

1.  **Agent**: The orchestrator that manages the interaction loop between the user, the LLM, and the tools.
2.  **Tool**: A wrapper around Python functions with argument validation and JSON schema generation.
3.  **ChatOpenAI**: A wrapper around the `openai` client that handles API communication, including `bind_tools`.

### Design Philosophy

- **Type Safety**: Heavily relies on Python type hints and Pydantic for validation.
- **Transparency**: Exposes raw OpenAI events and responses where possible.
- **Simplicity**: Minimal abstractions; "it's just Python functions".

---

## Execution Modes

LiteRun fully supports both **Synchronous** and **Asynchronous** (`asyncio`) environments. The API follows a consistent naming convention:

| Environment | Single Response            | Streaming Response                  |
| :---------- | :------------------------- | :---------------------------------- |
| **Sync**    | `agent.invoke(...)`        | `for x in agent.stream(...)`        |
| **Async**   | `await agent.ainvoke(...)` | `async for x in agent.astream(...)` |

This applies to both the `Agent` and the `ChatOpenAI`.

---

## Agent Reference

The `Agent` class orchestrates the conversation loop, including model calls, tool execution, and conversation state management.

### Parameters

- **`llm`** (`ChatOpenAI`) — The language model instance used by the agent.
- **`tools`** (`list[Tool]`, optional) — List of `Tool` instances available to the agent.
- **`system_prompt`** (`str`, optional) — System-level instruction provided to the model.
- **`tool_choice`** (`str`, optional) — Strategy for selecting tools.
  - `"auto"`: The model decides whether to call a tool or generate text.
  - `"required"`: The model _must_ call a tool. Useful for extraction tasks.
  - `"none"`: The model _cannot_ call tools. Useful for pure chat.
- **`parallel_tool_calls`** (`bool`, optional) — Whether to allow parallel execution of tools.
  - **Note**: While the API supports parallel tool calls, LiteRun currently runs the tools sequentially.
- **`max_iterations`** (`int`, optional) — Safety loop limit (default: 20).
  - Prevents infinite loops if the model keeps calling tools without giving a final answer.
  - Raises `RuntimeError` if exceeded.

### Execution Loop

The `invoke` (or `ainvoke`) method runs the following loop:

1.  **Input**: User text is added to history.
2.  **LLM Call**: Model generates response or tool calls.
3.  **Tool Execution**:
    - If tool calls are present, they are executed (sync or async).
    - Results are added to history.
    - Loop continues (Step 2).
4.  **Final Answer**: Returns final text if model completes.

### Return Object

Both `invoke` and `ainvoke` return a structured `RunResult` object, which provides:

- **`input`** (`str`) — The original user input to the agent.
- **`final_output`** (`str`) — The final text response from the agent.
- **`new_items`** (`list[RunItem]`) — Complete trace of the agent conversation turn.

### Example Usage

**1. Synchronous (Standard)**

```python
# 1. Invoke
result = agent.invoke(user_input="Calculate 5 * 5")
print(f"Answer: {result.final_output}")

# Inspecting the trace
for item in result.new_items:
    if item.type == "tool_call_item":
        print(f"Tool Called: {item.raw_item.name}")
    elif item.type == "tool_call_output_item":
        print(f"Tool Output: {item.content}")

# 2. Stream
for item in agent.stream(user_input="Hello"):
    if item.event.type == "response.output_text.delta":
        print(item.event.delta, end="")
```

**2. Asynchronous (`asyncio`)**

```python
# 1. Async Invoke
result = await agent.ainvoke(user_input="Hello")

# 2. Async Streaming
async for result in agent.astream(user_input="Hello"):
    event = result.event
    # process event...
```

---

## Tool Management

Tools are defined using the `Tool` class. You must provide a name, description, and the Python implementation.

### Tool Definition

A **Tool** represents a function the agent can call. Tools can be:

1. **Async** (`async def`) → use `coroutine=`
2. **Sync** (`def`) → use `func=` (executed directly in sync mode or in a thread pool in async mode)

### Parameters

- **`name`** (`str`) — Unique name for the tool.
- **`description`** (`str`) — What the tool does.
- **`func`** (`Callable`, optional) — Synchronous Python function.
- **`coroutine`** (`Callable`, optional) — Asynchronous Python function.
- **`strict`** (`bool`, optional) — Enforce strict JSON schema validation.
- **`args_schema`** (`list[ArgsSchema]`, optional) — Define argument names, types, and descriptions.

### Using `ArgsSchema`

The `ArgsSchema` maps argument names to types and descriptions. This generates the JSON Schema sent to OpenAI. Unlike automatic inspection, this gives you full control over the schema.

```python
from literun import Tool, ArgsSchema

def my_func(location: str, unit: str = "celsius"):
    return f"Weather in {location} is 25 {unit}"

tool = Tool(
    name="get_weather",
    description="Get weather for a location",
    func=my_func,
    args_schema=[
        ArgsSchema(
            name="location",
            type=str,
            description="City and state, e.g. San Francisco, CA"
        ),
        ArgsSchema(
            name="unit",
            type=str,
            description="Temperature unit",
            enum=["celsius", "fahrenheit"] # Restrict values
        )
    ]
)
```

### Async Compatibility & Threading

LiteRun intelligently handles execution mode mismatches:

1.  **Async Tools (`async def`)**:
    - Pass to `coroutine=` argument.
    - Executed directly via `await` in `ainvoke`/`astream`.
    - **Best for**: API calls, database queries, file I/O, network operations.
    - **Recommended** when using the `Agent` or `LLM` in **async** mode, to avoid blocking the event loop.

    ```python
    # 1. Async Tool (Native Async) - Example only, validate URLs in production
    async def fetch(url: str):
        # Note: In production, validate and sanitize URLs to prevent SSRF
        async with httpx.AsyncClient() as client:
            return await client.get(url)  # Add URL validation before use

    tool = Tool(name="fetch", coroutine=fetch, description="Fetch URL")
    ```

2.  **Sync Tools (`def`)**:
    - Pass to `func=` argument.
    - In `invoke` (Sync): Executed directly.
    - In `ainvoke` (Async): Executed in a **thread pool** (`asyncio.to_thread`) to prevent blocking the event loop.
    - Safe to use, but may incur a small overhead due to threading when using the `Agent` or `LLM` in **async** mode.

    ```python
    # 2. Sync Tool (Auto-threaded in async)
    def calculate(x: int):
        import time
        time.sleep(1) # This blocks, but it's safe in a thread!
        return x * 2

    tool = Tool(name="calc", func=calculate, description="Heavy calculation")
    ```

---

## ChatOpenAI Reference

A stateless wrapper around the OpenAI API. It handles client initialization, configuration, and request formatting.

### Parameters

- **`model`** (`str`) — The OpenAI model name to use (e.g., "gpt-4o").
- **`api_key`** (`str`, optional) — OpenAI API key. If not provided, takes from environment.
- **`temperature`** (`float`, optional) — Sampling temperature using default if not provided.
- **`organization`** (`str`, optional) — OpenAI organization ID.
- **`project`** (`str`, optional) — OpenAI project ID.
- **`base_url`** (`str`, optional) — Custom base URL for OpenAI API.
- **`max_output_tokens`** (`int`, optional) — Maximum tokens allowed in the model output.
- **`timeout`** (`float`, default=60) — Request timeout in seconds.
- **`max_retries`** (`int`, default=3) — Number of retries for failed requests.
- **`store`** (`bool`, optional) — Enable OpenAI storage.
- **`reasoning_effort`** (`str`, optional) — Level of reasoning effort by the model for reasoning models ("low"/"medium"/"high").
- **`verbosity`** (`str`, optional) — Level of verbosity in model responses ("low"/"medium"/"high").
- **`text_format`** (`str`, optional) — Format of the output text ("text"/"json_object"/"json_schema").
- **`model_kwargs`** (`dict[str, Any]`, default={}) — Additional model-specific parameters passed to OpenAI.

### Direct Usage (No Agent)

If you don't need the loop (tools -> execution -> loop), use the LLM directly.

### Example Usage

**1. Synchronous**

```python
# Bind tools
llm.bind_tools(
    tools=[my_tool],
    tool_choice="required",
    parallel_tool_calls=False
)

# Invoke
response = llm.invoke([{"role": "user", "content": "Hello"}])
print(response.output_text)

# Stream
for event in llm.stream([{"role": "user", "content": "Hello"}]):
    print(event)
```

**2. Asynchronous**

```python
# Async Invoke
response = await llm.ainvoke(...)

# Async Stream
async for event in llm.astream(...):
    print(event)
```

---

## Runtime Context Injection

Sometimes tools need access to data that shouldn't be visible to the LLM (e.g., database connections, User IDs, API keys). LiteRun supports **runtime context injection**.

1.  Annotate an argument in your tool function with `ToolRuntime`.
2.  Pass a dictionary to `agent.invoke(..., runtime_context={...})`.
3.  The agent will automatically strip this argument from the LLM schema and inject the context object at execution time.

**Example: Secure Database Access**

```python
from literun import ToolRuntime

def query_database(query: str, ctx: ToolRuntime) -> str:
    # Example only - demonstrates runtime context usage
    # In production: use parameterized queries, not raw SQL strings
    
    user_id = getattr(ctx, "user_id")
    db_conn = getattr(ctx, "db_connection")
    
    # Use parameterized queries in production to prevent SQL injection
    return db_conn.execute(query, user_id=user_id)

# Initialize tool (schema will ONLY show 'query')
tool = Tool(name="query_db", func=query_database, ...)

# Run agent with context
agent.invoke(
    "Get my latest orders",
    runtime_context={
        "user_id": 99,
        "db_connection": my_db_connection
    }
)
```

---

## Prompt Templates

The `PromptTemplate` class maintains conversation history and provides type-safe methods to add messages.

```python
from literun import PromptTemplate

template = PromptTemplate()

# 1. Add standard roles
template.add_system("You are a helpful assistant.")
template.add_user("Hello")
template.add_assistant("Hi there")

# 2. Add complex interactions (Tool calls/results)
# This is useful for restoring specific history states
template.add_tool_call(
    call_id="call_123",
    name="get_weather",
    arguments='{"location": "Tokyo"}'
)

template.add_tool_output(
    call_id="call_123",
    output="Sunny, 25C"
)

# 3. Use in execution
agent.invoke(user_input="How are you?", prompt_template=template)
```

---

## Streaming

Real-time streaming exposes granular events for UI updates. The stream yields `RunResultStreaming` objects containing an `event`.

### Key Event Types

| Event Type                                | Description                   | Content Field       |
| :---------------------------------------- | :---------------------------- | :------------------ |
| `response.output_text.delta`              | A token of generated text     | `event.delta`       |
| `response.function_call_arguments.delta`  | A fragment of JSON arguments  | `event.delta`       |
| `response.output_text.done`               | Text generation complete      | `event.text`        |
| `response.function_call_output_item.done` | A tool has finished executing | `event.item.output` |

### Full Async Streaming Loop

```python
async for result in agent.astream("Perform calculation"):
    event = result.event

    # 1. Text Streaming
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)

    # 2. Tool Arguments Streaming
    elif event.type == "response.function_call_arguments.delta":
        pass

    # 3. Tool Execution Finished
    elif event.type == "response.function_call_output_item.done":
        print(f"\n[Tool '{event.item.name}' returned: {event.item.output}]")

    # 4. Final Text Done
    elif event.type == "response.output_text.done":
        print("\n[Text Generation Complete]")
```

## Examples

For complete, runnable code examples covering these concepts, please visit the [**examples**](https://github.com/kaustubh-tr/literun/blob/main/examples/) directory in the repository.
