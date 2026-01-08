from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"

class ContentType(str, Enum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    MESSAGE = "message"  # NOTE: used only for incoming API responses; not valid for PromptMessage construction
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"

class StreamEventType(str, Enum):
    TEXT = "text"            # assistant text tokens
    TOOL_CALL = "tool_call"  # tool call + arguments
    LIFECYCLE = "lifecycle"  # started / completed 
    ERROR = "error"          # failed / incomplete / error 
    INTERNAL = "internal"    # raw passthrough (opt-in) 
    
class EventPhase(str, Enum):
    DELTA = "delta"
    FINAL = "final"
    NONE = "none"
    
class RunStatus(str, Enum):
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StreamStatus(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
