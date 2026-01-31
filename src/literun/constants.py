"""Constants and Enums for the literun package."""

from __future__ import annotations

from typing import Literal

Role = Literal["system", "user", "assistant", "developer", "tool"]

ContentType = Literal[
    "input_text",
    "output_text",
    "message",
    "function_call",
    "function_call_output",
]

ToolChoice = Literal["auto", "none", "required"]
ReasoningEffort = Literal["low", "medium", "high"]
Verbosity = Literal["low", "medium", "high"]
TextFormat = Literal["text", "json_object", "json_schema"]

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60.0  # seconds
DEFAULT_MAX_TOOL_CALLS_LIMIT = 10
DEFAULT_MAX_ITERATIONS_LIMIT = 20
