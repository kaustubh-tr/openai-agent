from typing import Any, Dict, List, Optional, Callable
from .args_schema import ArgsSchema


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

    # Runtime argument handling
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