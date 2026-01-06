from typing import Any, Dict, List, Optional, Generator
import json
from openai import OpenAI
from .tool import Tool
from .prompt_template import PromptTemplate
from .constants import ContentType, StreamEventType, EventPhase, ProcessStatus, StreamStatus
from .stream_event import StreamEvent


class Agent:
    """
    A minimal agent runtime built on OpenAI Responses API.
    """
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        parallel_tool_calls: Optional[bool] = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Agent.
        Args:
            model (str): The OpenAI model identifier (e.g., "gpt-4o").
            system_prompt (Optional[str]): The system instructions for the agent.
            parallel_tool_calls (Optional[bool]): Whether to allow parallel tool calls. Defaults to None (uses the OpenAI API's default behavior).
            max_iterations (int): Maximum number of agent loop iterations. Defaults to 10.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = max_iterations
        self.model_kwargs = kwargs

    # Tool registration
    def add_tool(self, tool: Tool) -> None:
        """
        Register a tool with the agent.
        Args:
            tool (Tool): The tool instance to register.
        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    # LLM call
    def _chat(self, messages: List[Dict[str, Any]], stream: bool = False) -> Any:
        """
        Internal method to call the OpenAI API.
        Args:
            messages (List[Dict[str, Any]]): The list of messages to send.
            stream (bool): Whether to stream the response.
        Returns:
            Any: The OpenAI API response (or stream).
        """
        params = {
            "model": self.model,
            "input": messages,
            "stream": stream,
            **self.model_kwargs,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.parallel_tool_calls is not None:
            params["parallel_tool_calls"] = self.parallel_tool_calls
        if self.tools:
            params["tools"] = [tool.to_openai_tool() for tool in self.tools.values()]
            params["tool_choice"] = "auto"
        return self.client.responses.create(**params)

    # Tool execution
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a registered tool.
        Args:
            name (str): The name of the tool to execute.
            args (Dict[str, Any]): The arguments for the tool.
        Returns:
            Any: The result of the tool execution.
        Raises:
            ValueError: If the tool is not found.
        """
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        parsed_args = tool.resolve_arguments(args)
        return tool.func(**parsed_args)
    
    # Initial message construction
    def _build_initial_prompt_messages(
        self, 
        user_input: str, 
        prompt_template: Optional[PromptTemplate] = None
    ) -> PromptTemplate:
        """
        Construct the initial conversation state.
        Args:
            user_input (str): The user's input text.
            prompt_template (Optional[PromptTemplate]): A template to initialize history.
        Returns:
            PromptTemplate: The constructed PromptTemplate instance.
        """
        # 1. Start with a PromptTemplate (message container)
        if prompt_template is not None:
            template = prompt_template.copy()
        else:
            template = PromptTemplate()
            if self.system_prompt:
                template.system(self.system_prompt)

        # 2. Add the user message using template.user()
        template.user(user_input)
        return template

    # Main agent loop
    def invoke(
        self,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None
    ) -> str:
        """
        Run the agent synchronously.
        Args:
            user_input (str): The user's input text.
            prompt_template (Optional[PromptTemplate]): Optional template for history.
        Returns:
            str: The final text response from the agent.
        Raises:
            ValueError: If user_input is empty.
            RuntimeError: If the agent exceeds max iterations or gets stuck.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")
        prompt_messages = self._build_initial_prompt_messages(user_input, prompt_template)

        for _ in range(self.max_iterations):
            response = self._chat(prompt_messages.to_openai_input())
            tool_called = False
            final_output: Optional[str] = None

            for item in response.output:
                # Tool call event
                if item.type == ContentType.FUNCTION_CALL:
                    tool_called = True
                    
                    # 1. Append tool call
                    prompt_messages.tool_call(
                        name=item.name,
                        arguments=item.arguments,
                        call_id=item.call_id,
                    )

                    # 2. Execute tool
                    raw_args = json.loads(item.arguments)
                    result = self._execute_tool(item.name, raw_args)

                    # 3. Append tool output
                    prompt_messages.tool_output(
                        call_id=item.call_id,
                        output=str(result),
                    )
                    
                # Final assistant message
                elif item.type == ContentType.MESSAGE:
                    # Extract text from content list
                    text_parts = [
                        c.text for c in item.content if c.type == ContentType.OUTPUT_TEXT
                    ]
                    final_output = "".join(text_parts)

            # If we processed tool calls, we loop again to get the next response.
            if tool_called:
                continue

            # If we didn't process tool calls, and we have a final output, we return it.
            if final_output is not None:
                return final_output
            
            # Safety break if no progress
            raise RuntimeError("Agent received neither a tool call nor a text message.")
        raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations})")
    
    # Main agent streaming loop
    def stream(
        self,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None,
        *,
        include_internal_events: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        """
        Run the agent with streaming responses.
        Args:
            user_input (str): The user's input text.
            prompt_template (Optional[PromptTemplate]): Optional template for history.
            include_internal_events (bool): Whether to emit raw internal events. Defaults to False.
        Yields:
            StreamEvent: Events representing the agent's progress and output.
        Raises:
            ValueError: If user_input is empty.
            RuntimeError: If the agent exceeds max iterations.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")
        prompt_messages = self._build_initial_prompt_messages(user_input, prompt_template)
        
        for _ in range(self.max_iterations):
            response_stream = self._chat(prompt_messages.to_openai_input(), stream=True)
            tool_calls: Dict[str, Dict[str, Any]] = {}
            tool_called = False

            for event in response_stream:
                if include_internal_events:
                    yield StreamEvent(
                        type=StreamEventType.INTERNAL,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        raw_event=event,
                    )
                
                if event.type == "response.created":
                    response = event.response
                    yield StreamEvent(
                        type=StreamEventType.LIFECYCLE,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        process_status=ProcessStatus.STARTED,
                        stream_status=StreamStatus.IDLE,
                        raw_event=event,
                    )
                    
                elif event.type == "response.output_item.added":
                    item = event.item
                    # Tool call created
                    if item.type == ContentType.FUNCTION_CALL:
                        tool_called = True
                        tool_calls[item.call_id] = {
                            "item_id": item.id,
                            "name": item.name,
                            "arguments": "",
                        }
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            phase=EventPhase.NONE,
                            sequence_number=event.sequence_number,
                            item_id=item.id,
                            tool_name=item.name,
                            call_id=item.call_id,
                            process_status=ProcessStatus.IN_PROGRESS,
                            stream_status=StreamStatus.STARTED,
                            raw_event=event,
                        )
                    
                    # Message stream created
                    elif item.type == ContentType.MESSAGE:
                        yield StreamEvent(
                            type=StreamEventType.TEXT,
                            phase=EventPhase.NONE,
                            sequence_number=event.sequence_number,
                            item_id=item.id,
                            process_status=ProcessStatus.IN_PROGRESS,
                            stream_status=StreamStatus.STARTED,
                            raw_event=event,
                        )

                # Tool arguments streaming
                elif event.type == "response.function_call_arguments.delta":
                    for call_id, call in tool_calls.items():
                        if call["item_id"] == event.item_id:
                            call["arguments"] += event.delta
                        
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                phase=EventPhase.DELTA,
                                sequence_number=event.sequence_number,
                                item_id=event.item_id,
                                tool_name=call["name"],
                                call_id=call_id,
                                arguments=event.delta,
                                process_status=ProcessStatus.IN_PROGRESS,
                                stream_status=StreamStatus.STREAMING,
                                raw_event=event,
                            )
                        
                # Tool arguments completed
                elif event.type == "response.function_call_arguments.done":
                    for call_id, call in tool_calls.items():
                        if call["item_id"] == event.item_id:
                            call["arguments"] = event.arguments
                        
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                phase=EventPhase.FINAL,
                                sequence_number=event.sequence_number,
                                item_id=event.item_id,
                                tool_name=call["name"],
                                call_id=call_id,
                                arguments=event.arguments,
                                process_status=ProcessStatus.IN_PROGRESS,
                                stream_status=StreamStatus.COMPLETED,
                                raw_event=event,
                            )
                        
                # Text streaming
                elif event.type == "response.output_text.delta":
                    yield StreamEvent(
                        type=StreamEventType.TEXT,
                        phase=EventPhase.DELTA,
                        sequence_number=event.sequence_number,
                        item_id=event.item_id,
                        text=event.delta,
                        process_status=ProcessStatus.IN_PROGRESS,
                        stream_status=StreamStatus.STREAMING,
                        raw_event=event,
                    )
                
                # Text completed
                elif event.type == "response.output_text.done":
                    yield StreamEvent(
                        type=StreamEventType.TEXT,
                        phase=EventPhase.FINAL,
                        sequence_number=event.sequence_number,
                        item_id=event.item_id,
                        text=event.text,
                        process_status=ProcessStatus.IN_PROGRESS,
                        stream_status=StreamStatus.COMPLETED,
                        raw_event=event,
                    )
                        
                # Response completed successfully
                elif event.type == "response.completed":
                    response = event.response
                    yield StreamEvent(
                        type=StreamEventType.LIFECYCLE,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        usage=response.usage,
                        process_status=ProcessStatus.COMPLETED,
                        stream_status=StreamStatus.COMPLETED,
                        raw_event=event,
                    )
                
                # Response incomplete
                elif event.type == "response.incomplete":
                    response = event.response
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        error=response.incomplete_details,  # {'reason': ...}
                        process_status=ProcessStatus.IN_PROGRESS,
                        stream_status=StreamStatus.INCOMPLETE,
                        raw_event=event,
                    )
                    return
                
                # Response failed (doesn't have sequence_number)
                elif event.type == "response.failed":
                    response = event.response
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        response_id=response.id,
                        error=response.error,  # {'code': ..., 'message': ...}
                        process_status=ProcessStatus.FAILED,
                        stream_status=StreamStatus.FAILED,
                        raw_event=event,
                    )
                    return
                
                # Error (doesn't have response_id)
                elif event.type == "error":
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        error={"code": event.code, "message": event.message},  # {'code': ..., 'message': ...}
                        process_status=ProcessStatus.FAILED,
                        stream_status=StreamStatus.FAILED,
                        raw_event=event,
                    )
                    return
            
            # No tool calls -> agent is done
            if not tool_called: return
            
            # Execute tools and append results to prompt messages
            for call_id, call in tool_calls.items():
                prompt_messages.tool_call(
                    name=call["name"],
                    arguments=call["arguments"],
                    call_id=call_id,
                )
                result = self._execute_tool(
                    call["name"],
                    json.loads(call["arguments"]),
                )
                prompt_messages.tool_output(
                    call_id=call_id,
                    output=str(result),
                )

        raise RuntimeError("Streaming agent exceeded max iterations")
    