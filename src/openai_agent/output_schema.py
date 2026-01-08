from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .constants import StreamEventType, EventPhase, RunStatus, StreamStatus


@dataclass(frozen=True)
class Response:
    """
    Final result returned by the OpenAI Agent.
    Used in Agent.invoke() method.
    """
    output: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Any] = None
    status: Optional[str] = None
    raw_response: Optional[Any] = None


@dataclass(frozen=True)
class ResponseStreamEvent:
    """
    Represents a streaming event from the agent or LLM.
    Used for streaming responses from Agent.stream() and ChatOpenAI.stream().
    """
    # Classification
    type: StreamEventType
    phase: Optional[EventPhase] = None
    
    # Ordering
    sequence_number: Optional[int] = None
    
    # Identity
    response_id: Optional[str] = None
    item_id: Optional[str] = None
    
    # Payload (for streaming)
    text: Optional[str] = None
    tool_name: Optional[str] = None
    call_id: Optional[str] = None
    arguments: Optional[str] = None
    usage: Optional[dict] = None
    
    # Lifecycle
    run_status: Optional[RunStatus] = None
    stream_status: Optional[StreamStatus] = None
    
    # Errors / raw passthrough
    error: Optional[Dict[str, Any]] = None
    raw_event: Optional[Any] = None
