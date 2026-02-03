"""Return types for agent execution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import Any

from openai.types.responses import Response
from .items import RunItem
from .events import StreamEvent


class RunResult(BaseModel):
    """Final result returned by the OpenAI Agent.

    Used in the ``Agent.invoke()`` method. Contains the full execution
    trace accumulated during a single agent run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: str | list[Any]
    """The original input items before ``run()`` was called."""

    new_items: Response | list[RunItem]
    """Items generated during the agent run, such as messages, tool calls,
    and tool outputs.
    """

    final_output: Any
    """The output produced by the final agent invocation."""


class RunResultStreaming(BaseModel):
    """Streaming result returned by the OpenAI Agent.

    Used in the ``Agent.stream()`` method. Each instance represents a
    single stream event while accumulating completed items.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input: str | list[Any]
    """The original input items before ``run()`` was called."""

    event: StreamEvent
    """The stream event emitted for this iteration."""

    final_output: Any
    """The output produced by the final agent invocation.

    This value is `None` until the final message is complete.
    """
