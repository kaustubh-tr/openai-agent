"""LLM client wrapper and configuration."""

from __future__ import annotations

from typing import Any, Iterator
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from openai import OpenAI
from openai.types.responses import Response

from .tool import Tool
from .prompt_template import PromptTemplate
from .events import ResponseStreamEvent
from .constants import (
    Verbosity,
    TextFormat,
    ReasoningEffort,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_MODEL,
)


class ChatOpenAI(BaseModel):
    """Stateless wrapper for a configured OpenAI model.

    Provides a unified interface to call the OpenAI Responses API, optionally
    binding tools and streaming outputs.

    Args:
        model: The model name to use.
        temperature: Sampling temperature.
        api_key: OpenAI API key.
        organization: OpenAI organization ID.
        project: OpenAI project ID.
        base_url: Custom base URL for OpenAI API.
        max_output_tokens: Maximum number of tokens in the output.
        timeout: Request timeout in seconds.
        max_retries: Number of retries for failed requests.
        reasoning_effort: Level of reasoning effort for the model.
            Options: "none", "low", "medium", "high".
        verbosity: Level of verbosity in model responses.
            Options: "low", "medium", "high".
        text_format: Format of the text output.
            Options: "text", "json_object", "json_schema".
        store: Whether to store model responses with OpenAI.
        model_kwargs: Additional model parameters.
    """

    model: str = DEFAULT_OPENAI_MODEL
    temperature: float | None = None
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | None = None
    max_output_tokens: int | None = None
    timeout: float | None = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    reasoning_effort: ReasoningEffort | None = None
    verbosity: Verbosity | None = None
    text_format: TextFormat | None = None
    store: bool = False
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    _client: OpenAI = PrivateAttr()
    _tools: list[Tool] | None = PrivateAttr(default=None)
    _tool_choice: str | None = PrivateAttr(default=None)
    _parallel_tool_calls: bool | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_temperature(self) -> ChatOpenAI:
        """Validate temperature and reasoning parameters."""
        model_lower = self.model.lower()

        # 1. For o-series models, default to temperature=1 if not provided
        if model_lower.startswith("o") and self.temperature is None:
            import warnings

            warnings.warn(
                "o-series models require temperature=1 and no temperature was provided. "
                "Setting default temperature=1 for o-series models.",
                UserWarning,
            )
            self.temperature = 1

        # 2. For gpt-5 models, handle temperature restrictions
        # (Assuming gpt-5-chat and non-reasoning gpt-5 support arbitrary temps)
        is_gpt5 = model_lower.startswith("gpt-5")
        is_chat = "chat" in model_lower

        # Check reasoning effort from field or model_kwargs
        effort_kwarg = (self.model_kwargs.get("reasoning") or {}).get("effort")
        has_reasoning = (self.reasoning_effort and self.reasoning_effort != "none") or (
            effort_kwarg and effort_kwarg != "none"
        )

        if is_gpt5 and not is_chat and has_reasoning:
            if self.temperature is not None and self.temperature != 1:
                import warnings

                warnings.warn(
                    "Invalid temperature for gpt-5 with reasoning. Using default temperature.",
                    UserWarning,
                )
                self.temperature = None

        return self

    @model_validator(mode="after")
    def _initialize_client(self) -> ChatOpenAI:
        """Initialize the OpenAI client."""
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self

    @property
    def client(self) -> OpenAI:
        """Access the OpenAI client."""
        return self._client

    def bind_tools(
        self,
        *,
        tools: list[Tool],
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> ChatOpenAI:
        """Bind tools to the LLM instance.

        Args:
            tools: List of Tool instances to bind.
            tool_choice: Optional tool selection strategy.
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            ``ChatOpenAI``: The updated instance with tools bound.
        """
        self._tools = tools
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls
        return self

    def chat(
        self,
        *,
        messages: PromptTemplate | list[dict[str, Any]],
        stream: bool = False,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> Response | Iterator[ResponseStreamEvent]:
        """Call the model with the given messages.

        Args:
            messages: PromptTemplate or list of messages in OpenAI format.
            stream: Whether to stream the output.
            tools: Optional list of Tool instances.
            tool_choice: Optional tool selection strategy.
            parallel_tool_calls: Whether to allow parallel tool calls.

        Returns:
            Response | Iterator[ResponseStreamEvent]: The OpenAI Responses API response object (or stream).
        """
        if isinstance(messages, PromptTemplate):
            input_ = messages.to_openai_input()
        else:
            input_ = messages

        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "input": input_,
            "stream": stream,
            "store": self.store,
            **self.model_kwargs,
        }
        if self.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.reasoning_effort}

        text_options = {}
        if self.verbosity is not None:
            text_options["verbosity"] = self.verbosity
        if self.text_format is not None:
            text_options["type"] = self.text_format
        if text_options:
            params["text"] = {"format": text_options}

        # Tools resolution
        active_tools = tools if tools is not None else self._tools
        current_tools = (
            self._convert_to_openai_tools(active_tools) if active_tools else None
        )

        if current_tools:
            params["tools"] = current_tools
            params["tool_choice"] = tool_choice or self._tool_choice
            params["parallel_tool_calls"] = (
                parallel_tool_calls
                if parallel_tool_calls is not None
                else self._parallel_tool_calls
            )

        return self.client.responses.create(**params)

    def invoke(self, messages: list[dict[str, Any]] | PromptTemplate) -> Response:
        """Synchronously call the model.

        Args:
            messages: PromptTemplate or list of messages in OpenAI format.

        Returns:
            Response: The OpenAI Responses API response object.
        """
        return self.chat(messages=messages, stream=False)

    def stream(
        self,
        *,
        messages: list[dict[str, Any]] | PromptTemplate,
    ) -> Iterator[ResponseStreamEvent]:
        """Stream the model response.

        Args:
            messages: PromptTemplate or list of messages in OpenAI format.

        Yields:
            ResponseStreamEvent: Streamed response events from the OpenAI Responses API.
        """
        response = self.chat(messages=messages, stream=True)
        for event in response:
            yield event

    @staticmethod
    def _convert_to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]] | None:
        """Convert all registered tools to the OpenAI tool schema format.

        Returns:
            List[Dict[str, Any]]: A list of tools in OpenAI-compatible dictionary format.
        """
        if not tools:
            return None
        return [tool.to_openai_tool() for tool in tools]
