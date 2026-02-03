"""Tool definition and runtime context."""

from __future__ import annotations

import inspect
import asyncio
from typing import Any, get_type_hints
from collections.abc import Awaitable, Callable
from pydantic import BaseModel, ConfigDict, model_validator

from .args_schema import ArgsSchema


class ToolRuntime(BaseModel):
    """Runtime context container for tools.

    This class corresponds to arbitrary runtime values passed via `runtime_context`
    in `Agent.invoke()`. It allows extra arguments on initialization.

    Example:
        runtime = ToolRuntime(user_id="user123", session="abc")
        print(runtime.user_id)  # user123
    """

    model_config = ConfigDict(extra="allow")


class Tool(BaseModel):
    """Represents a callable tool that can be invoked by an agent or LLM.

    A ``Tool`` wraps a Python callable along with metadata and an argument
    schema, and provides utilities for argument validation, execution,
    and conversion to the OpenAI tool definition format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    """The name of the tool."""

    description: str = ""
    """A description of what the tool does."""

    func: Callable[..., str] | None = None
    """The function to run when the tool is called."""

    coroutine: Callable[..., Awaitable[str]] | None = None
    """The asynchronous version of the function."""

    args_schema: list[ArgsSchema] | None = None
    """The schema of the tool's input arguments."""

    strict: bool | None = None
    """If True, model output is guaranteed to exactly match the JSON Schema
    provided in the function definition. If None, `strict` argument will not
    be included in tool definition."""

    @model_validator(mode="after")
    def _validate_callable(self) -> Tool:
        """Ensure correct usage of func (sync) vs coroutine (async)."""
        if self.func is None and self.coroutine is None:
            raise ValueError("One of `func` or `coroutine` must be provided.")

        # func must be synchronous (not a coroutine)
        if self.func and inspect.iscoroutinefunction(self.func):
            raise ValueError("`func` should be a synchronous function, not async.")

        # coroutine must be an async function
        if self.coroutine and not inspect.iscoroutinefunction(self.coroutine):
            raise ValueError("`coroutine` should be an async function.")

        return self

    @model_validator(mode="after")
    def _validate_name(self) -> Tool:
        """Ensure tool has a valid name."""
        if not self.name:
            if self.func:
                self.name = self.func.__name__
            elif self.coroutine:
                self.name = self.coroutine.__name__
            else:
                raise ValueError("Tool must have a name or a callable with a name.")
        return self

    # LLM Runtime argument handling
    def _resolve_arguments(self, raw_args: dict[str, Any]) -> dict[str, Any]:
        """Validate and cast raw arguments provided by the model.

        Args:
            raw_args: The raw argument dictionary produced by the model.

        Returns:
            dict[str, Any]: A dictionary of validated and type-cast arguments.
        """
        parsed: dict[str, Any] = {}
        if self.args_schema:
            for arg in self.args_schema:
                parsed[arg.name] = arg.validate_and_cast(raw_args.get(arg.name))
        else:
            parsed = raw_args or {}
        return parsed

    def _inject_runtime(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None,
        target_callable: Callable[..., Any],
    ) -> dict[str, Any]:
        """Inject `ToolRuntime` if the callable requests it."""
        final_args = dict(args)

        try:
            type_hints = get_type_hints(target_callable)
        except Exception:
            sig = inspect.signature(target_callable)
            type_hints = {
                name: param.annotation
                for name, param in sig.parameters.items()
                if param.annotation != inspect.Parameter.empty
            }

        for pname, ptype in type_hints.items():
            if ptype is ToolRuntime:
                final_args[pname] = ToolRuntime(**(runtime_context or {}))

        return final_args

    def run(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the tool synchronously."""
        if not self.func:
            raise RuntimeError("This tool has no synchronous implementation")
        parsed_args = self._resolve_arguments(args)
        final_args = self._inject_runtime(parsed_args, runtime_context, self.func)
        return self.func(**final_args)

    async def arun(
        self,
        args: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute the tool asynchronously.

        If `coroutine` is provided, it is used. Otherwise, `func` is run
        in a thread pool to avoid blocking the event loop.
        """
        parsed_args = self._resolve_arguments(args)
        if self.coroutine:
            final_args = self._inject_runtime(
                parsed_args, runtime_context, self.coroutine
            )
            return await self.coroutine(**final_args)

        # Fallback: run sync func on a thread
        final_args = self._inject_runtime(parsed_args, runtime_context, self.func)
        return await asyncio.to_thread(self.func, **final_args)

    def convert_to_openai_tool(self) -> dict[str, Any]:
        """Convert the tool to the OpenAI tool schema format.

        Returns:
            dict[str, Any]: The OpenAI-compatible tool definition.
        """
        properties = {}
        required = []

        if self.args_schema:
            for arg in self.args_schema:
                properties[arg.name] = arg.convert_to_json_schema()
                required.append(arg.name)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            **({"strict": self.strict} if self.strict is not None else {}),
        }
