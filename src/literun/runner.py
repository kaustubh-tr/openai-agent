"""Agent execution runner."""

from __future__ import annotations

import json
from typing import Any, Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

from .items import (
    RunItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    ReasoningItem,
    ResponseFunctionToolCallOutput,
)
from .results import RunResult, RunResultStreaming
from .events import (
    ResponseFunctionCallOutputItemAddedEvent,
    ResponseFunctionCallOutputItemDoneEvent,
)
from .prompt import PromptTemplate


class Runner:
    """Executes agent runs."""

    @classmethod
    def run(
        cls,
        agent: Agent,
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

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations`.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = cls._build_prompt(agent, user_input, prompt_template)
        all_items: list[RunItem] = []

        iteration = 0
        while iteration < agent.max_iterations:
            response = agent.llm.chat(
                messages=prompt,
                stream=False,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            tool_calls: dict[str, dict[str, Any]] = {}
            final_output_text: str = ""

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

                tool_output = cls._run_tool(agent, name, arguments_str, runtime_context)

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
            iteration += 1

        raise RuntimeError(f"Agent exceeded max iterations ({agent.max_iterations})")

    @classmethod
    async def arun(
        cls,
        agent: Agent,
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

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations`.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = cls._build_prompt(agent, user_input, prompt_template)
        all_items: list[RunItem] = []

        iteration = 0
        while iteration < agent.max_iterations:
            response = await agent.llm.achat(
                messages=prompt,
                stream=False,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            tool_calls: dict[str, dict[str, Any]] = {}
            final_output_text: str = ""

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

                tool_output = await cls._arun_tool(
                    agent, name, arguments_str, runtime_context
                )

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
            iteration += 1

        raise RuntimeError(f"Agent exceeded max iterations ({agent.max_iterations})")

    @classmethod
    def run_stream(
        cls,
        agent: Agent,
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

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations`.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = cls._build_prompt(agent, user_input, prompt_template)

        iteration = 0
        while iteration < agent.max_iterations:
            response_stream = agent.llm.chat(
                messages=prompt,
                stream=True,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            tool_calls: dict[str, dict[str, Any]] = {}
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

                tool_output = cls._run_tool(agent, name, arguments_str, runtime_context)

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
            iteration += 1

        raise RuntimeError(f"Agent exceeded max iterations ({agent.max_iterations})")

    @classmethod
    async def arun_stream(
        cls,
        agent: Agent,
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

        Raises:
            ValueError: If `user_input` is empty.
            RuntimeError: If the agent exceeds `max_iterations`.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")

        prompt = cls._build_prompt(agent, user_input, prompt_template)

        iteration = 0
        while iteration < agent.max_iterations:
            response_stream = await agent.llm.achat(
                messages=prompt,
                stream=True,
                tools=agent.tools,
                tool_choice=agent.tool_choice,
                parallel_tool_calls=agent.parallel_tool_calls,
            )

            tool_calls: dict[str, dict[str, Any]] = {}
            final_output_text: str = ""

            async for event in response_stream:
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

                tool_output = await cls._arun_tool(
                    agent, name, arguments_str, runtime_context
                )

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
            iteration += 1

        raise RuntimeError(f"Agent exceeded max iterations ({agent.max_iterations})")

    @staticmethod
    def _build_prompt(
        agent: Agent,
        user_input: str,
        prompt_template: PromptTemplate | None = None,
    ) -> PromptTemplate:
        """Construct the conversation state for a new agent turn.

        Args:
            user_input: The user's input text.
            prompt_template: Optional template to initialize conversation history.
                If None, a new ``PromptTemplate`` is created, and the system prompt is added if available.

        Returns:
            ``PromptTemplate``: The fully constructed prompt containing system, user, and previous messages.
        """
        if prompt_template is not None:
            prompt = prompt_template.copy()
        else:
            prompt = PromptTemplate()
            if agent.system_prompt:
                prompt.add_system(agent.system_prompt)

        prompt.add_user(user_input)
        return prompt

    @staticmethod
    def _run_tool(
        agent: Agent,
        name: str,
        arguments: str | dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
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
        tool = agent._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            result = tool.run(args, runtime_context)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    @staticmethod
    async def _arun_tool(
        agent: Agent,
        name: str,
        arguments: str | dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a registered tool asynchronously safely with provided arguments.

        Args:
            name: The name of the tool to execute.
            arguments: Arguments to pass to the tool, either as a JSON string or dict.
            runtime_context: Optional runtime context to pass to tool arguments of type ``ToolRuntime``.

        Returns:
            str: The output of the tool execution, or an error message if execution fails.
        """
        tool = agent._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments

            result = await tool.arun(args, runtime_context)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"
