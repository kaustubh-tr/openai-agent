from __future__ import annotations
from .prompt_message import PromptMessage
from typing import Iterable, List, Dict, Any
from .constants import Role, ContentType


class PromptTemplate:
    """
    Container for conversation state.

    This is the authoritative message history used by the Agent.
    It stores PromptMessage objects and serializes them only at the
    OpenAI API boundary.
    """
    
    def __init__(self) -> None:
        """Initialize an empty PromptTemplate."""
        self.messages: List[PromptMessage] = []
        
    def add_message(self, message: PromptMessage) -> PromptTemplate:
        """
        Add a custom PromptMessage.
        Args:
            message (PromptMessage): The message to add.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        Raises:
            TypeError: If `message` is not an instance of PromptMessage.
        """
        if not isinstance(message, PromptMessage):
            raise TypeError("Expected PromptMessage")
        self.messages.append(message)
        return self
    
    def add_messages(self, messages: Iterable[PromptMessage]) -> PromptTemplate:
        """
        Add multiple PromptMessages.
        Args:
            messages (Iterable[PromptMessage]): The messages to add.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        for message in messages:
            self.add_message(message)
        return self

    def system(self, text: str) -> PromptTemplate:
        """
        Add a system message.
        Args:
            text (str): The system message text.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.DEVELOPER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )

    def user(self, text: str) -> PromptTemplate:
        """
        Add a user message.
        Args:
            text (str): The user message text.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.USER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )

    def assistant(self, text: str) -> PromptTemplate:
        """
        Add an assistant message.
        Args:
            text (str): The assistant message text.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                role=Role.ASSISTANT,
                content_type=ContentType.OUTPUT_TEXT,
                text=text,
            )
        )

    def tool_call(
        self,
        *,
        name: str,
        arguments: str,
        call_id: str,
    ) -> PromptTemplate:
        """
        Add a tool call message.
        Args:
            name (str): The name of the tool called.
            arguments (str): The arguments passed to the tool (as a JSON string).
            call_id (str): The unique ID of the tool call.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL,
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
        )

    def tool_output(
        self,
        *,
        call_id: str,
        output: str,
    ) -> PromptTemplate:
        """
        Add a tool output message.
        Args:
            call_id (str): The ID of the tool call this output corresponds to.
            output (str): The output of the tool execution.
        Returns:
            PromptTemplate: The current instance, allowing method chaining.
        """
        return self.add_message(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL_OUTPUT,
                call_id=call_id,
                output=output,
            )
        )
    
    def copy(self) -> PromptTemplate:
        """
        Create a shallow copy of this template.
        Required to avoid mutating caller-owned templates inside Agent.
        Returns:
            PromptTemplate: A new PromptTemplate instance with the same messages.
        """
        new = PromptTemplate()
        new.messages = list(self.messages)
        return new
    
    def to_openai_input(self) -> List[Dict[str, Any]]:
        """
        Convert the template to a list of OpenAI message dictionaries.
        Returns:
            List[Dict[str, Any]]: The list of formatted messages.
        """
        return [msg.to_openai_message() for msg in self.messages]
    
    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)