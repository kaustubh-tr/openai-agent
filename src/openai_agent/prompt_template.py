from .prompt_message import PromptMessage
from typing import List, Dict, Any
from .constants import Role, ContentType


class PromptTemplate:
    def __init__(self):
        self.messages: List[PromptMessage] = []

    # ------------------------
    # Add blocks
    # ------------------------
    def system(self, text: str):
        self.messages.append(
            PromptMessage(
                role=Role.DEVELOPER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )
        return self

    def user(self, text: str):
        self.messages.append(
            PromptMessage(
                role=Role.USER,
                content_type=ContentType.INPUT_TEXT,
                text=text,
            )
        )
        return self

    def assistant(self, text: str):
        self.messages.append(
            PromptMessage(
                role=Role.ASSISTANT,
                content_type=ContentType.OUTPUT_TEXT,
                text=text,
            )
        )
        return self

    def tool_call(
        self,
        *,
        name: str,
        arguments: str,
        call_id: str,
    ):
        self.messages.append(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL,
                name=name,
                arguments=arguments,
                call_id=call_id,
            )
        )
        return self

    def tool_output(
        self,
        *,
        call_id: str,
        output: str,
    ):
        self.messages.append(
            PromptMessage(
                content_type=ContentType.FUNCTION_CALL_OUTPUT,
                call_id=call_id,
                output=output,
            )
        )
        return self
    
    def to_openai_input(self) -> List[Dict[str, Any]]:
        return [msg.to_openai_message() for msg in self.messages]