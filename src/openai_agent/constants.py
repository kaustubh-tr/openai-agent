from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"

class ContentType(str, Enum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"

