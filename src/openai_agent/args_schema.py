from typing import Any, List, Optional, Type


class Arg:
    def __init__(
        self,
        name: str,
        arg_type: Type,
        description: str,
        enum: Optional[List[Any]] = None,
    ):
        self.name = name
        self.arg_type = arg_type
        self.description = description
        self.enum = enum

    # ----------------------------
    # JSON schema representation
    # ----------------------------
    def to_json_schema(self) -> dict:
        schema = {
            "type": self._json_type(),
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        return schema

    # ----------------------------
    # Runtime validation / coercion
    # ----------------------------
    def validate_and_cast(self, value: Any) -> Any:
        if value is None:
            raise ValueError(f"Missing required argument '{self.name}'")

        try:
            return self.arg_type(value)
        except Exception as e:
            raise ValueError(
                f"Invalid value for '{self.name}': expected {self.arg_type.__name__}"
            ) from e

    # ----------------------------
    # Helpers
    # ----------------------------
    def _json_type(self) -> str:
        if self.arg_type is str:
            return "string"
        if self.arg_type is int:
            return "integer"
        if self.arg_type is float:
            return "number"
        if self.arg_type is bool:
            return "boolean"
        raise ValueError(f"Unsupported type: {self.arg_type}")