from dataclasses import dataclass
from typing import Optional, Dict, Any
from .constants import StreamEventType, EventPhase, ProcessStatus, StreamStatus


@dataclass(frozen=True)
class StreamEvent:
    # Classification
    type: StreamEventType
    phase: EventPhase
    
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
    process_status: Optional[ProcessStatus] = None
    stream_status: Optional[StreamStatus] = None
    
    # Errors / raw passthrough
    error: Optional[Dict[str, Any]] = None
    raw_event: Optional[Any] = None
