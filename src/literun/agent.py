"""Agent runtime implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Iterator, List, Union

from .tool import Tool
from .llm import ChatOpenAI
from .prompt_template import PromptTemplate
from .results import RunResult, RunResultStreaming
from .items import (
    RunItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
    ResponseFunctionToolCallOutput,
)
from .events import (
    StreamEvent,
    ResponseFunctionCallOutputItemAddedEvent,
    ResponseFunctionCallOutputItemDoneEvent,
)


class Agent:
    """A minimal agent runtime built on OpenAI Responses API."""

    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: str = "auto",  # "auto", "none", "required"
        parallel_tool_calls: bool = True,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the Agent.

        Args:
            llm: The OpenAI language model instance to use.
            system_prompt: The system instructions for the agent.
            tools: An optional list of Tool instances to register.
            tool_choice: Strategy for selecting tools during execution.
                One of "auto", "none", or "required".
            parallel_tool_calls: Whether to call tools in parallel.
            max_iterations: Maximum number of iterations for the agent loop. Must be >= 1.

        Raises:
            ValueError: If max_iterations is less than 1.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.max_iterations = max_iterations
        self.tools: Dict[str, Tool] = self.add_tools(tools)

    def add_tools(
        self,
        tools: Optional[Iterable[Tool]],
    ) -> Dict[str, Tool]:
        """Register a set of tools for the agent.

        Args:
            tools: An optional list of Tool instances to register.

        Returns:
            Dict[str, Tool]: A mapping from tool names to their Tool instances.

        Raises:
            ValueError: If there are duplicate tool names.
        """
        tool_map: Dict[str, Tool] = {}
        for tool in tools or []:
            if tool.name in tool_map:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            tool_map[tool.name] = tool
        return tool_map

    def add_tool(self, tool: Tool) -> None:
        """Add a single tool at runtime.

        This method MUTATES agent state. Intended for advanced/dynamic use cases.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def _convert_to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all registered tools to the OpenAI tool schema format.

        Returns:
            List[Dict[str, Any]]: A list of tools in OpenAI-compatible dictionary format.
        """
        return [tool.to_openai_tool() for tool in self.tools.values()]

    def _execute_tool(
        self,
        name: str,
        arguments: Union[str, Dict[str, Any]],
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a registered tool safely with provided arguments.

        Handles parsing of arguments (from JSON string or dict) and catches execution errors.

        Args:
            name: The name of the tool to execute.
            arguments: Arguments to pass to the tool, either as a JSON string or dict.
            runtime_context: Optional runtime context to pass to tool arguments of type ``ToolRuntime``.

        Returns:
            str: The output of the tool execution, or an error message if execution fails.
        """
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            result = tool.execute(args, runtime_context)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    def _build_prompt(
        self, user_input: str, prompt_template: Optional[PromptTemplate] = None
    ) -> PromptTemplate:
        """Construct the conversation state for a new agent turn.

        Args:
            user_input: The user's input text.
            prompt_template: Optional template to initialize the conversation history.
                If None, a new ``PromptTemplate`` is created, and the system prompt is added if available.

        Returns:
            ``PromptTemplate``: The fully constructed prompt containing system, user, and previous messages.
        """
        if prompt_template is not None:
            prompt = prompt_template.copy()
        else:
            prompt = PromptTemplate()
            if self.system_prompt:
                prompt.add_system(self.system_prompt)

        prompt.add_user(user_input)
        return prompt

    def invoke(
        self,
        *,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> RunResult:
        """Run the agent synchronously.

        This method executes the agent loop, calling the language model and any
        registered tools until a final output is produced or the maximum number
        of iterations is reached. Each step in the execution is recorded in
        the returned ``RunResult``.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Returns:
            ``RunResult``: Contains the original input, all items generated
                during execution (messages, tool calls, reasoning), and the final output.

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations` without completing.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = self._build_prompt(user_input, prompt_template)
        all_items: List[RunItem] = []

        for _ in range(self.max_iterations):
            response = self.llm.chat(
                messages=prompt.to_openai_input(),
                stream=False,
                tools=self._convert_to_openai_tools() if self.tools else None,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            )

            tool_calls: Dict[str, Dict[str, Any]] = {}
            final_output_text: str = ""

            # Process each output item from OpenAI response
            for item in response.output:
                if item.type == "reasoning":
                    all_items.append(
                        ReasoningItem(
                            role="assistant",
                            content=item.content,
                            raw_item=item,
                            type="reasoning_item",
                        )
                    )

                elif item.type == "function_call":
                    tool_calls[item.id] = {
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    }
                    all_items.append(
                        ToolCallItem(
                            role="assistant",
                            content="",
                            raw_item=item,
                            type="tool_call_item",
                        )
                    )

                elif item.type == "message":
                    text_parts = [
                        c.text for c in item.content if c.type == "output_text"
                    ]
                    final_output_text = "".join(text_parts)
                    all_items.append(
                        MessageOutputItem(
                            role="assistant",
                            content=final_output_text,
                            raw_item=item,
                            type="message_output_item",
                        )
                    )

            if not tool_calls:
                return RunResult(
                    input=user_input,
                    new_items=all_items,
                    final_output=final_output_text,
                )

            # Update history with assistant's text (Critical for context)
            if final_output_text:
                prompt.add_assistant(final_output_text)

            for tc in tool_calls.values():
                call_id = tc["call_id"]
                name = tc["name"]
                arguments_str = tc["arguments"]

                prompt.add_tool_call(
                    name=name,
                    arguments=arguments_str,
                    call_id=call_id,
                )

                tool_output = self._execute_tool(name, arguments_str, runtime_context)

                prompt.add_tool_output(call_id=call_id, output=tool_output)

                all_items.append(
                    ToolCallOutputItem(
                        role="tool",
                        content=tool_output,
                        raw_item=ResponseFunctionToolCallOutput(
                            call_id=call_id,
                            output=tool_output,
                            name=name,
                            type="function_call_output",
                            status="completed",
                        ),
                        type="tool_call_output_item",
                    )
                )

        raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations})")

    def stream(
        self,
        *,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[RunResultStreaming]:
        """Run the agent with streaming output.

        This method streams response events from the agent as they occur,
        including messages, tool calls, and tool outputs. It allows
        real-time processing of the agent's reasoning and tool execution.

        Args:
            user_input: The input text from the user.
            prompt_template: Optional template to initialize conversation history.
            runtime_context: Optional runtime context dictionary to pass to tools.

        Yields:
            ``RunResultStreaming``: Streaming events containing the current input,
                the event from the LLM or tool, and the accumulated final output.

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations`.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = self._build_prompt(user_input, prompt_template)

        for _ in range(self.max_iterations):
            response_stream = self.llm.chat(
                messages=prompt.to_openai_input(),
                stream=True,
                tools=self._convert_to_openai_tools() if self.tools else None,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            )

            tool_calls: Dict[str, Dict[str, Any]] = {}
            final_output_text: str = ""

            for event in response_stream:
                yield RunResultStreaming(
                    input=user_input,
                    event=event,
                    final_output=final_output_text,
                )

                if event.type == "response.output_item.done":
                    if event.item.type == "message":
                        for content_part in event.item.content:
                            if content_part.type == "output_text":
                                final_output_text += content_part.text

                    elif event.item.type == "function_call":
                        tool_calls[event.item.id] = {
                            "call_id": event.item.call_id,
                            "name": event.item.name,
                            "arguments": event.item.arguments,
                        }

            if not tool_calls:
                return

            # Update history with assistant's text (Critical for context)
            if final_output_text:
                prompt.add_assistant(final_output_text)

            for tc in tool_calls.values():
                call_id = tc["call_id"]
                name = tc["name"]
                arguments_str = tc["arguments"]

                prompt.add_tool_call(
                    name=name, arguments=arguments_str, call_id=call_id
                )

                yield RunResultStreaming(
                    input=user_input,
                    event=ResponseFunctionCallOutputItemAddedEvent(
                        type="response.function_call_output_item.added",
                        item=ResponseFunctionToolCallOutput(
                            call_id=call_id,
                            output="",
                            name=name,
                            type="function_call_output",
                            status="in_progress",
                        ),
                        output_index=None,
                        sequence_number=None,
                    ),
                    final_output=final_output_text,
                )

                tool_output = self._execute_tool(name, arguments_str, runtime_context)

                prompt.add_tool_output(call_id=call_id, output=tool_output)

                yield RunResultStreaming(
                    input=user_input,
                    event=ResponseFunctionCallOutputItemDoneEvent(
                        type="response.function_call_output_item.done",
                        item=ResponseFunctionToolCallOutput(
                            call_id=call_id,
                            output=tool_output,
                            name=name,
                            type="function_call_output",
                            status="completed",
                        ),
                        output_index=None,
                        sequence_number=None,
                    ),
                    final_output=final_output_text,
                )

        raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations})")
