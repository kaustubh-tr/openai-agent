from __future__ import annotations
from typing import Any, Dict, List, Optional, Generator
from openai import OpenAI
from .tool import Tool
from .constants import StreamEventType, EventPhase
from .output_schema import Response, ResponseStreamEvent
from .utils import extract_output_text, extract_tool_calls, extract_usage_dict

class ChatOpenAI:
    """
    A Stateless configured OpenAI model wrapper.
    """
    def __init__(
        self,
        *,
        model: str = "gpt-4.1",
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        store: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ChatOpenAI instance.
        Args:
            model (str): The model name to use.
            temperature (Optional[float]): Sampling temperature.
            api_key (Optional[str]): OpenAI API key.
            base_url (Optional[str]): Custom base URL for OpenAI API.
            max_output_tokens (Optional[int]): Maximum tokens in the output.
            tools (Optional[List[Tool]]): List of Tool instances to bind.
            tool_choice (Optional[str]): Tool choice strategy.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
            **kwargs: Additional model parameters.
        """
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.model_kwargs = kwargs
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )
        self.store = store
        self._tools = tools
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls

    def bind_tools(
        self,
        *,
        tools: List[Tool],
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ChatOpenAI:
        """
        Bind tools to the LLM instance.
        Args:
            tools (List[Tool]): List of Tool instances to bind.
            tool_choice (Optional[str]): Tool choice strategy.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls.
        Returns:
            ChatOpenAI: The updated LLM instance with tools bound.
        """
        self._tools = tools
        self._tool_choice = tool_choice
        self._parallel_tool_calls = parallel_tool_calls
        return self

    def _chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> Any:
        """
        The unified interface for calling the model.
        Args:
            messages: List of messages in OpenAI format.
            stream: Stream the output if True.
        Returns:
            Any: The OpenAI API response (or stream).
        """
        params = {
            "model": self.model,
            "input": messages,
            "stream": stream,
            "store": self.store,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            params["max_output_tokens"] = self.max_output_tokens
        
        # Tools resolution
        current_tools = tools or (
            [tool.to_openai_tool() for tool in self._tools] if self._tools else None
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
        
    def invoke(self, messages: List[Dict[str, Any]]) -> Response:
        """
        Synchronously call the model.
        Args:
            messages: List of messages.
        Returns:
            Response: The OpenAI API response wrapped in Response.
        """
        response = self._chat(messages=messages, stream=False)
        output_text = extract_output_text(response)
        tool_calls = extract_tool_calls(response)
        usage_dict = extract_usage_dict(response)

        return Response(
            output=output_text,
            tool_calls=tool_calls,
            usage=usage_dict,
            status=getattr(response, "status", None),
            raw_response=response
        )

    def stream(
        self,
        *,
        messages: List[Dict[str, Any]],
        include_internal_events: bool = False
    ) -> Generator[ResponseStreamEvent, None, None]:
        """
        Stream the model response.
        Args:
            messages: List of messages.
            include_internal_events: Whether to emit raw internal events.
        Yields:
            ResponseStreamEvent: Streaming events.
        """
        response_stream = self._chat(messages=messages, stream=True)
        tool_calls: Dict[str, Dict[str, Any]] = {}

        for event in response_stream:
            if include_internal_events:
                yield ResponseStreamEvent(
                    type=StreamEventType.INTERNAL,
                    phase=EventPhase.NONE,
                    raw_event=event,
                )

            # Tool call created
            if event.type == "response.output_item.added":
                item = event.item
                if item.type == "function_call":
                    tool_calls[item.id] = {
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": "",
                    }
                    yield ResponseStreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        phase=EventPhase.NONE,
                        tool_name=item.name,
                        call_id=item.call_id,
                        raw_event=event,
                    )

            # Tool arguments streaming
            elif event.type == "response.function_call_arguments.delta":
                call = tool_calls.get(event.item_id)
                if call:
                    call["arguments"] += event.delta
                    yield ResponseStreamEvent(
                        type=StreamEventType.TOOL_CALL,
                        phase=EventPhase.DELTA,
                        tool_name=call["name"],
                        call_id=call["call_id"],
                        arguments=event.delta,
                        raw_event=event,
                    )
                    
            # Text streaming
            elif event.type == "response.output_text.delta":
                yield ResponseStreamEvent(
                    type=StreamEventType.TEXT,
                    phase=EventPhase.DELTA,
                    text=event.delta,
                    raw_event=event,
                )
                    
            # Response completed successfully
            elif event.type == "response.completed":
                response = event.response
                yield ResponseStreamEvent(
                    type=StreamEventType.LIFECYCLE,
                    phase=EventPhase.FINAL,
                    response_id=response.id,
                    usage=extract_usage_dict(response),
                    raw_event=event,
                )
    