from typing import Any, List, Optional, Type, Dict


class Arg:
    """
    Represents an argument for a tool.
    """
    def __init__(
        self,
        name: str,
        arg_type: Type,
        description: str,
        enum: Optional[List[Any]] = None,
    ):
        """
        Initialize an Arg.
        Args:
            name (str): The name of the argument.
            arg_type (Type): The Python type of the argument (e.g., str, int).
            description (str): A description of the argument.
            enum (Optional[List[Any]]): A list of allowed values for the argument.
        """
        self.name = name
        self.type_ = arg_type
        self.description = description
        self.enum = enum

    # JSON schema representation
    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert the argument to a JSON schema dictionary.
        Returns:
            dict: The JSON schema representation of the argument.
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
        """
        Validate and cast a value to the argument's type.
        Args:
            value (Any): The value to validate and cast.
        Returns:
            Any: The cast value.
        Raises:
            ValueError: If the value is missing or invalid for the type.
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
        """
        Get the JSON type string corresponding to the Python type.
        Returns:
            str: The JSON type string (e.g., "string", "integer").
        Raises:
            ValueError: If the Python type is not supported.
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