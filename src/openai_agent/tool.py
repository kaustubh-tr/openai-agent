from typing import Any, Dict, List, Optional, Callable, get_type_hints
import inspect
from .args_schema import ArgsSchema


class ToolRuntime:
    """
    Base class for runtime context.
    Attributes match the keys of the context dictionary passed to the agent.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f"ToolRuntime({self.__dict__})"


class Tool:
    """
    Represents a tool that the agent can use.
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
        """
        Initialize a Tool.
        Args:
            func (Callable): The function to execute when the tool is called.
            name (str): The name of the tool.
            description (str): A description of what the tool does.
            args_schema (List[ArgsSchema]): A list of arguments the tool accepts.
            strict (Optional[bool]): If True, model output is guaranteed to exactly match the JSON Schema
                provided in the function definition. If None, ``strict`` argument will not
                be included in tool definition.
        """
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.strict = strict

    # OpenAI schema
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert the tool to the OpenAI tool schema.
        Returns:
            Dict[str, Any]: The OpenAI tool definition.
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
        """
        Validate and cast arguments from the model.
        Args:
            raw_args (Dict[str, Any]): The raw arguments from the model.
        Returns:
            Dict[str, Any]: The validated and cast arguments.
        """
        parsed = {}
        for arg in self.args_schema:
            parsed[arg.name] = arg.validate_and_cast(raw_args.get(arg.name))
        return parsed

    def execute(self, args: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the tool with the given arguments and runtime context.
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
                param_name: param.annotation 
                for param_name, param in sig.parameters.items()
                if param.annotation != inspect.Parameter.empty
            }
        
        for param_name, param_type in type_hints.items():
            if param_type is ToolRuntime:
                final_args[param_name] = ToolRuntime(**(runtime_context or {}))

        return self.func(**final_args)
    