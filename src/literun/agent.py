"""Agent runtime implementation."""

from __future__ import annotations

from typing import Any, Iterator, AsyncIterator
from pydantic import BaseModel, PrivateAttr, model_validator

from .tool import Tool
from .llm import ChatOpenAI
from .prompt import PromptTemplate
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
    """

    llm: ChatOpenAI
    """The language model used by the agent."""
    
    system_prompt: str | None = None
    """The system prompt to initialize the agent's behavior."""
    
    tools: list[Tool] | None = None
    """A list of tools available to the agent."""
    
    tool_choice: ToolChoice = "auto"
    """The strategy for selecting tools during execution. 

    Options: `auto`, `none`, `required`.
    """
    
    parallel_tool_calls: bool = True
    """Whether to call tools in parallel when multiple are invoked."""
    
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS_LIMIT
    """Maximum number of tool calls allowed per agent run."""
    
    max_iterations: int = DEFAULT_MAX_ITERATIONS_LIMIT
    """Maximum number of iterations for the agent loop."""

    _tools: dict[str, Tool] = PrivateAttr(default_factory=dict)
    """Internal mapping of tool names to Tool instances."""

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
            ``RunResult``: The result of the agent run.
        """
        return Runner.run(
            agent=self,
            user_input=user_input,
            prompt_template=prompt_template,
            runtime_context=runtime_context,
        )

    async def ainvoke(
        self,
        *,
        user_input: str,
        prompt_template: PromptTemplate | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> RunResult:
        """Run the agent asynchronously.

        Executes the agent loop asynchronously, calling the language model
        and tools until a final response is produced.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Returns:
            ``RunResult``: The result of the agent run.
        """
        return await Runner.arun(
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
        """Run the agent synchronously with streaming output.

        Streams events as they occur (tokens, tool calls, tool results).
        Useful for real-time user interfaces.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Yields:
            ``RunResultStreaming``: Individual streaming events from the agent execution.
        """
        return Runner.run_stream(
            agent=self,
            user_input=user_input,
            prompt_template=prompt_template,
            runtime_context=runtime_context,
        )

    async def astream(
        self,
        *,
        user_input: str,
        prompt_template: PromptTemplate | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[RunResultStreaming]:
        """Run the agent asynchronously with streaming output.

        Streams events asynchronously as they occur (tokens, tool calls, tool results).
        Useful for real-time user interfaces.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Yields:
            ``RunResultStreaming``: Individual streaming events from the agent execution.
        """
        async for event in Runner.arun_stream(
            agent=self,
            user_input=user_input,
            prompt_template=prompt_template,
            runtime_context=runtime_context,
        ):
            yield event
