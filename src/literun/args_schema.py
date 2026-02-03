"""Schema definition for tool arguments."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ArgsSchema(BaseModel):
    """Represents an argument for a tool."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    """The name of the argument."""
    
    type_: type[Any] = Field(alias="type")
    """The Python type of the argument."""
    
    description: str = ""
    """A description of the argument."""
    
    enum: list[Any] | None = None
    """List of allowed values for the argument."""

    # JSON schema representation
    def convert_to_json_schema(self) -> dict[str, Any]:
        """Convert the argument to a JSON Schema representation.

        Returns:
            dict[str, Any]: A dictionary representing the argument in JSON Schema format.
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
