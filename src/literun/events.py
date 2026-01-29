"""Response events for streaming agent execution."""

from __future__ import annotations

from typing import Union, Literal, TypeAlias
from pydantic import BaseModel

from .items import ResponseFunctionToolCallOutput
from openai.types.responses import (
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseQueuedEvent,
    ResponseCreatedEvent,
    ResponseCompletedEvent,
    ResponseTextDoneEvent,
    ResponseIncompleteEvent,
    ResponseTextDeltaEvent,
    ResponseInProgressEvent,
    ResponseRefusalDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseReasoningTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
)


ResponseStreamEvent: TypeAlias = Union[
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseQueuedEvent,
]


# Custom internal events
class ResponseFunctionCallOutputItemAddedEvent(BaseModel):
    """Emitted when a function tool call output item is created."""

    item: ResponseFunctionToolCallOutput
    """The output item that was added."""

    output_index: None
    """The index of the output item that was added. Always `None` for function tool call output."""

    sequence_number: None
    """The sequence number of this event. Always `None` for function tool call output."""

    type: Literal["response.function_call_output_item.added"]
    """The type of the event. Always `response.function_call_output_item.added`."""


class ResponseFunctionCallOutputItemDoneEvent(BaseModel):
    """Emitted when a function tool call output item is marked done."""

    item: ResponseFunctionToolCallOutput
    """The output item that was marked done."""

    output_index: None
    """The index of the output item that was added. Always `None` for function tool call output."""

    sequence_number: None
    """The sequence number of this event. Always `None` for function tool call output."""

    type: Literal["response.function_call_output_item.done"]
    """The type of the event. Always `response.function_call_output_item.done`."""


TResponseStreamEvent = ResponseStreamEvent

StreamEvent: TypeAlias = Union[
    TResponseStreamEvent,
    ResponseFunctionCallOutputItemAddedEvent,
    ResponseFunctionCallOutputItemDoneEvent,
]
