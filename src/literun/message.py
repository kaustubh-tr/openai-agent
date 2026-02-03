"""Message structures for prompts."""

from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel, model_validator

from .constants import Role, ContentType


class PromptMessage(BaseModel):
    """Domain representation of a single semantic message in a conversation.

    This class is the only place that knows how to convert a semantic
    message into an OpenAI-compatible message dictionary. It enforces
    invariants depending on the message type.
    """

    role: Role | None = None
    """The role of the message sender.

    Options: `system`, `user`, `assistant`, `tool`
    """

    text: str | None = None
    """The text content of the message."""

    name: str | None = None
    """The name of the tool."""

    arguments: str | None = None
    """The arguments for the tool as a JSON string."""

    call_id: str | None = None
    """The call ID of the tool call."""

    output: str | None = None
    """The output of the tool execution."""

    content_type: ContentType
    """The type of content.

    Options: `text`, `tool_call`, `tool_call_output`
    """

    @model_validator(mode="after")
    def _validate_invariants(self) -> PromptMessage:
        """Enforce invariants so that invalid messages are never constructed.

        Raises:
            ValueError: If required fields are missing for the given content_type.
        """
        # Text messages (system / user / assistant)
        if self.content_type == "text":
            if self.role is None:
                raise ValueError("role is required for text messages")
            if not isinstance(self.text, str):
                raise ValueError("text is required for text messages")

        # Tool call (model -> agent)
        elif self.content_type == "tool_call":
            if not self.name:
                raise ValueError("name is required for tool_call")
            if not isinstance(self.arguments, str):
                raise ValueError("arguments must be a JSON string")
            if not self.call_id:
                raise ValueError("call_id is required for tool_call")

        # Tool output (agent -> model)
        elif self.content_type == "tool_call_output":
            if not self.call_id:
                raise ValueError("call_id is required for tool_call_output")
            if not isinstance(self.output, str):
                raise ValueError("output must be a string")
        else:
            raise ValueError(f"Unsupported content_type: {self.content_type}")

        return self

    def convert_to_openai_message(self) -> Dict[str, Any]:
        """Convert the PromptMessage to an OpenAI-compatible message dictionary.

        Returns:
            Dict[str, Any]: The formatted message dictionary.

        Raises:
            ValueError: If required fields are missing for the specified content_type.
            RuntimeError: If the message state is invalid (should not occur).
        """
        # System / User / Assistant messages
        if self.content_type == "text":
            if self.role == "system":
                return {
                    "role": self.role,
                    "content": [
                        {"type": "input_text", "text": self.text},
                    ],
                }
            elif self.role == "user":
                return {
                    "role": self.role,
                    "content": [
                        {"type": "input_text", "text": self.text},
                    ],
                }
            elif self.role == "assistant":
                return {
                    "role": self.role,
                    "content": [
                        {"type": "output_text", "text": self.text},
                    ],
                }
            else:
                raise ValueError(f"Unsupported role: {self.role}")
            
        # Tool call
        if self.content_type == "tool_call":
            return {
                "type": "function_call",
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

        # Tool output
        if self.content_type == "tool_call_output":
            return {
                "type": "function_call_output",
                "call_id": self.call_id,
                "output": self.output,
            }

        # Should never reach here due to validation
        raise ValueError("Invalid PromptMessage state")
