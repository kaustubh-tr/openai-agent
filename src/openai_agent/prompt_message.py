from typing import Any, Dict, List, Optional
from .constants import Role, ContentType


class PromptMessage:
    def __init__(
        self,
        *,
        role: Optional[Role] = None,
        content_type: ContentType,
        text: Optional[str] = None,
        name: Optional[str] = None,
        arguments: Optional[str] = None,
        call_id: Optional[str] = None,
        output: Optional[str] = None,
    ):
        self.role = role
        self.content_type = content_type
        self.text = text
        self.name = name
        self.arguments = arguments
        self.call_id = call_id
        self.output = output
        
    def to_openai_message(self) -> Dict[str, Any]:
        # System / User / Assistant messages
        if self.content_type in (ContentType.INPUT_TEXT, ContentType.OUTPUT_TEXT):
            if self.role is None:
                raise ValueError("role is required for text messages")
            if self.text is None:
                raise ValueError("text is required for text messages")

            return {
                "role": self.role.value,
                "content": [
                    {
                        "type": self.content_type.value,
                        "text": self.text,
                    }
                ],
            }

        # Tool call (model -> agent)
        if self.content_type == ContentType.FUNCTION_CALL:
            if self.name is None or self.arguments is None or self.call_id is None:
                raise ValueError(
                    "FUNCTION_CALL requires name, arguments, and call_id"
                )
                
            return {
                "type": self.content_type.value,
                "name": self.name,
                "arguments": self.arguments,
                "call_id": self.call_id,
            }

        # Tool output (agent -> model)
        if self.content_type == ContentType.FUNCTION_CALL_OUTPUT:
            if self.call_id is None or self.output is None:
                raise ValueError(
                    "FUNCTION_CALL_OUTPUT requires call_id and output"
                )
            
            return {
                "type": self.content_type.value,
                "call_id": self.call_id,
                "output": self.output,
            }

        raise ValueError(f"Unsupported content_type: {self.content_type}")
