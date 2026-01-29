"""Template management for constructing prompts."""

from __future__ import annotations

from typing import Iterable, List, Dict, Any

from .prompt_message import PromptMessage
from .constants import Role, ContentType


class PromptTemplate:
    """Container for conversation state.

    This class stores the authoritative message history used by the Agent.
    It manages ``PromptMessage`` objects and serializes them only at the
    OpenAI API boundary.
    """

    def __init__(self) -> None:
        """Initialize an empty PromptTemplate."""

        self.messages: List[PromptMessage] = []

    def add_message(self, message: PromptMessage) -> PromptTemplate:
        """Add a custom prompt message.

        Args:
            message: The message to add.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.

        Raises:
            TypeError: If `message` is not a ``PromptMessage`` instance.
        """
        if not isinstance(message, PromptMessage):
            raise TypeError("Expected PromptMessage")
        self.messages.append(message)
        return self

    def add_messages(self, messages: Iterable[PromptMessage]) -> PromptTemplate:
        """Add multiple prompt messages.

        Args:
            messages: An iterable of messages to add.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        for message in messages:
            self.add_message(message)
        return self

    def add_system(self, text: str) -> PromptTemplate:
        """Add a system message.

        Args:
            text: The system message text.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.DEVELOPER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )

    def add_user(self, text: str) -> PromptTemplate:
        """Add a user message.

        Args:
            text: The user message text.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.USER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )

    def add_assistant(self, text: str) -> PromptTemplate:
        """Add an assistant message.

        Args:
            text: The assistant message text.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.ASSISTANT,
                content_type=ContentType.OUTPUT_TEXT,
                text=text,
            )
        )

    def add_tool_call(
        self,
        *,
        name: str,
        arguments: str,
        call_id: str,
    ) -> PromptTemplate:
        """Add a tool call message.

        Args:
            name: The name of the tool being called.
            arguments: The tool arguments, encoded as a JSON string.
            call_id: The unique identifier for this tool call.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL,
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
        )

    def add_tool_output(
        self,
        *,
        call_id: str,
        output: str,
    ) -> PromptTemplate:
        """Add a tool output message.

        Args:
            call_id: The ID of the tool call this output corresponds to.
            output: The output produced by the tool.

        Returns:
            ``PromptTemplate``: The template instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL_OUTPUT,
                call_id=call_id,
                output=output,
            )
        )

    def copy(self) -> PromptTemplate:
        """Create a shallow copy of this template.

        This is required to avoid mutating caller-owned templates inside
        the Agent.

        Returns:
            ``PromptTemplate``: A new template containing the same messages.
        """
        new = PromptTemplate()
        new.messages = list(self.messages)
        return new

    def to_openai_input(self) -> List[Dict[str, Any]]:
        """Convert the template to OpenAI message dictionaries.

        Returns:
            List[Dict[str, Any]]: The formatted messages.
        """
        return [msg.to_openai_message() for msg in self.messages]

    def __len__(self) -> int:
        """Return the number of messages in the template."""
        return len(self.messages)

    def __iter__(self):
        """Iterate over stored messages."""
        return iter(self.messages)
