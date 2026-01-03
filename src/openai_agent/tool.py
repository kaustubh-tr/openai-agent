from typing import Any, Dict, List
from .args_schema import Arg


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        args: List[Arg],
        func,
        strict: bool = True,
    ):
        self.name = name
        self.description = description
        self.args = args
        self.func = func
        self.strict = strict

    # ----------------------------
    # OpenAI schema
    # ----------------------------
    def to_openai_tool(self) -> Dict[str, Any]:
        properties = {}
        required = []

        for arg in self.args:
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
            "strict": self.strict,
        }

    # ----------------------------
    # Runtime argument handling
    # ----------------------------
    def resolve_arguments(self, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        parsed = {}

        for arg in self.args:
            parsed[arg.name] = arg.validate_and_cast(raw_args.get(arg.name))

        return parsed