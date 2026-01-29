"""Tool definition and runtime context."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Callable, get_type_hints

from .args_schema import ArgsSchema


class ToolRuntime:
    """Runtime context container for tools.

    This class stores arbitrary runtime values as attributes, using keyword
    arguments provided at initialization. It is typically injected into tool
    functions that declare a parameter annotated with ``ToolRuntime``.

    Args:
        **kwargs: Arbitrary keyword arguments that will be set as attributes
            on the instance.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"ToolRuntime({self.__dict__})"


class Tool:
    """Represents a callable tool that can be invoked by an agent or LLM.

    A ``Tool`` wraps a Python callable along with metadata and an argument
    schema, and provides utilities for argument validation, execution,
    and conversion to the OpenAI tool definition format.
    """

    def __init__(
        self,
        *,
        func: Callable,
        name: str,
        description: str,
        args_schema: List[ArgsSchema],
        strict: Optional[bool] = None,
    ):
        """Initialize a Tool.

        Args:
            func: The function to execute when the tool is called.
            name: The name of the tool.
            description: A description of what the tool does.
            args_schema: A list of arguments the tool accepts.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the function definition. If None, `strict` argument will not
                be included in tool definition.
        """
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.strict = strict

    # OpenAI schema
    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert the tool to the OpenAI tool schema format.

        Returns:
            Dict[str, Any]: The OpenAI-compatible tool definition.
        """
        properties = {}
        required = []

        for arg in self.args_schema:
            properties[arg.name] = arg.to_json_schema()
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

    # LLM Runtime argument handling
    def resolve_arguments(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and cast raw arguments provided by the model.

        Args:
            raw_args: The raw argument dictionary produced by the model.

        Returns:
            Dict[str, Any]: A dictionary of validated and type-cast arguments.
        """
        parsed = {}
        for arg in self.args_schema:
            parsed[arg.name] = arg.validate_and_cast(raw_args.get(arg.name))
        return parsed

    def execute(
        self, args: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool with validated arguments and runtime context.

        This method resolves and validates model-provided arguments, injects
        a ``ToolRuntime`` instance into the function call if requested by the
        function's type annotations, and then invokes the underlying function.

        Args:
            args: Raw arguments provided by the model.
            runtime_context: Optional runtime context data used to construct
                a ``ToolRuntime`` instance when required.

        Returns:
            Any: The return value of the underlying tool function.
        """
        # 1. Resolve LLM arguments using the tool's schema logic
        final_args = self.resolve_arguments(args)

        # 2. Inject ToolRuntime if requested by the function signature
        # Use get_type_hints to properly resolve annotations, including forward references
        try:
            type_hints = get_type_hints(self.func)
        except (NameError, AttributeError, TypeError):
            # Fallback to inspect.signature if get_type_hints fails
            # This handles cases where annotations can't be resolved
            sig = inspect.signature(self.func)
            type_hints = {
                name: param.annotation
                for name, param in sig.parameters.items()
                if param.annotation != inspect.Parameter.empty
            }

        for param_name, param_type in type_hints.items():
            if param_type is ToolRuntime:
                final_args[param_name] = ToolRuntime(**(runtime_context or {}))

        return self.func(**final_args)
