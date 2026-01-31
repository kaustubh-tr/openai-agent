"""Agent runtime implementation."""

from __future__ import annotations

from typing import Any, Iterator
from pydantic import BaseModel, PrivateAttr, model_validator

from .tool import Tool
from .llm import ChatOpenAI
from .prompt_template import PromptTemplate
from .results import RunResult, RunResultStreaming
from .runner import Runner
from .constants import (
    ToolChoice,
    DEFAULT_MAX_TOOL_CALLS_LIMIT,
    DEFAULT_MAX_ITERATIONS_LIMIT,
)


class Agent(BaseModel):
    """A minimal agent runtime built on OpenAI Responses API.

    This class holds the configuration and state of the agent.
    Execution logic is delegated to the `Runner` class.

    Args:
        llm: The OpenAI language model instance to use.
        system_prompt: The system instructions for the agent.
        tools: An optional list of Tool instances to register.
        tool_choice: Strategy for selecting tools during execution.
            Options: "auto", "none", "required".
        parallel_tool_calls: Whether to call tools in parallel.
        max_iterations: Maximum number of iterations for the agent loop. Must be >= 1.

    Raises:
        ValueError: If max_iterations is less than 1.
    """

    llm: ChatOpenAI
    system_prompt: str | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool = True
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_LIMIT
    max_iterations: int = DEFAULT_MAX_ITERATIONS_LIMIT

    _tools: dict[str, Tool] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_config(self) -> Agent:
        """Validate configuration and initialize tools."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        return self

    @model_validator(mode="after")
    def _initialize_tools(self) -> Agent:
        self._tools = self.add_tools(self.tools)
        # Convert a list of tools to internal dictionary
        return self

    def add_tools(
        self,
        tools: list[Tool] | None,
    ) -> dict[str, Tool]:
        """Register a set of tools for the agent.

        Args:
            tools: An optional list of Tool instances to register.

        Returns:
            dict[str, Tool]: A mapping from tool names to their Tool instances.

        Raises:
            ValueError: If there are duplicate tool names.
        """
        tool_map: dict[str, Tool] = {}
        for tool in tools or []:
            if tool.name in tool_map:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_map[tool.name] = tool
        return tool_map

    def add_tool(self, tool: Tool) -> None:
        """Add a single tool at runtime.

        This method mutates agent state (internal tool registry).
        Intended for advanced/dynamic use cases.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

        # Keep public list in sync for Runner/LLM usage
        if self.tools is None:
            self.tools = []
        self.tools.append(tool)

    def invoke(
        self,
        *,
        user_input: str,
        prompt_template: PromptTemplate | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent synchronously.

        Executes the agent loop, calling the language model and tools until
        a final response is generated.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Returns:
            ``RunResult``: The result of the agent run, including user input,
                internal items (messages, tool calls), and the final output text.
        """
        return Runner.run(
            agent=self,
            user_input=user_input,
            prompt_template=prompt_template,
            runtime_context=runtime_context,
        )

    def stream(
        self,
        *,
        user_input: str,
        prompt_template: PromptTemplate | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> Iterator[RunResultStreaming]:
        """Run the agent with streaming output.

        Streams events as they happen (tokens, tool calls, tool results).
        Useful for real-time UIs.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Yields:
            ``RunResultStreaming``: Iteration of events from the agent execution.
        """
        return Runner.run_streamed(
            agent=self,
            user_input=user_input,
            prompt_template=prompt_template,
            runtime_context=runtime_context,
        )
