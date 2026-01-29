"""Schema definition for tool arguments."""

from __future__ import annotations

from typing import Any, List, Optional, Type, Dict


class ArgsSchema:
    """Represents an argument for a tool."""

    def __init__(
        self,
        *,
        name: str,
        type: Type,
        description: str,
        enum: Optional[List[Any]] = None,
    ):
        """Initialize an ArgsSchema.

        Args:
            name: The name of the argument.
            type: The Python type of the argument (e.g., str, int, float, bool).
            description: A description of the argument for documentation purposes.
            enum: Optional list of allowed values for the argument.
        """
        self.name = name
        self.type_ = type
        self.description = description
        self.enum = enum

    # JSON schema representation
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert the argument to a JSON Schema representation.

        Returns:
            Dict[str, Any]: A dictionary representing the argument in JSON Schema format.
        """
        schema = {
            "type": self._json_type(),
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema

    # Runtime validation / coercion
    def validate_and_cast(self, value: Any) -> Any:
        """Validate a value against the argument's type and cast it.

        Args:
            value: The value to validate and cast.

        Returns:
            Any: The value cast to the argument's Python type.

        Raises:
            ValueError: If the value is missing or cannot be cast to the expected type.
        """
        if value is None:
            raise ValueError(f"Missing required argument '{self.name}'")
        try:
            return self.type_(value)
        except Exception as e:
            raise ValueError(
                f"Invalid value for '{self.name}': expected {self.type_.__name__}"
            ) from e

    # Helpers
    def _json_type(self) -> str:
        """Get the JSON Schema type corresponding to the Python type.

        Returns:
            str: The JSON type string (e.g., "string", "integer", "number", "boolean").

        Raises:
            ValueError: If the Python type is unsupported.
        """
        if self.type_ is str:
            return "string"
        if self.type_ is int:
            return "integer"
        if self.type_ is float:
            return "number"
        if self.type_ is bool:
            return "boolean"
        raise ValueError(f"Unsupported type: {self.type_}")
