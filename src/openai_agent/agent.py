from typing import Any, Dict, Optional, Generator
import json
from .tool import Tool
from .llm import ChatOpenAI
from .prompt_template import PromptTemplate
from .constants import StreamEventType, EventPhase, RunStatus, StreamStatus
from .output_schema import Response, ResponseStreamEvent
from .utils import extract_usage_dict


class Agent:
    """
    A minimal agent runtime built on OpenAI Responses API.
    """
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        system_prompt: Optional[str] = None,
        tool_choice: str = "auto", # "auto", "none", "required"
        parallel_tool_calls: bool = True,
        max_iterations: int = 10,
    ) -> None:
        """
        Initialize the Agent.
        Args:
            llm (ChatOpenAI): The OpenAI language model instance to use.
            system_prompt (Optional[str]): The system instructions for the agent.
            tool_choice (str): Tool choice strategy: "auto", "none", "required".
            parallel_tool_calls (bool): Whether to call tools in parallel.
            max_iterations (int): Maximum number of agent loop iterations. Defaults to 10.
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.max_iterations = max_iterations
        self.tools: Dict[str, Tool] = {}

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
        
    def _convert_to_openai_tools(self):
        """
        Convert registered tools to OpenAI tool schema.
        Returns:
            List[Dict[str, Any]]: List of OpenAI tool definitions.
        """
        return [tool.to_openai_tool() for tool in self.tools.values()]

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
        final_args = tool.resolve_arguments(args)
        return tool.func(**final_args)
    
    # Initial message construction
    def _build_initial_prompt(
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
            prompt = prompt_template.copy()
        else:
            prompt = PromptTemplate()
            if self.system_prompt:
                prompt.system(self.system_prompt)

        # 2. Add the user message using prompt.user()
        prompt.user(user_input)
        return prompt

    # Main agent loop
    def invoke(
        self,
        *,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None
    ) -> Response:
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
        prompt = self._build_initial_prompt(user_input, prompt_template)

        for _ in range(self.max_iterations):
            response = self.llm._chat(
                messages=prompt.to_openai_input(),
                stream=False,
                tools=self._convert_to_openai_tools() if self.tools else None,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            )
            tool_calls = []
            output_text = None

            for item in response.output:
                if item.type == "function_call":
                    tool_calls.append(item)
                    
                elif item.type == "message":
                    text_parts = [
                        c.text for c in item.content if c.type == "output_text"
                    ]
                    output_text = "".join(text_parts)

            # If we processed tool calls, we loop again to get the next response.
            if tool_calls:
                for tool in tool_calls:
                    prompt.tool_call(
                        name=tool.name,
                        arguments=tool.arguments,
                        call_id=tool.call_id,
                    )
                    args = json.loads(tool.arguments)
                    result = self._execute_tool(tool.name, args)
                    
                    prompt.tool_output(
                        call_id=tool.call_id,
                        output=str(result),
                    )
                continue

            if output_text is not None:
                usage_dict = extract_usage_dict(response)
                return Response(
                    output=output_text,
                    usage=usage_dict,
                    status=getattr(response, "status", None),
                    raw_response=response
                )
            
            # Safety break if no progress
            raise RuntimeError("Agent received neither a tool call nor a text message.")
        raise RuntimeError(f"Agent exceeded max iterations ({self.max_iterations})")
    
    # Main agent streaming loop
    def stream(
        self,
        *,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None,
        include_internal_events: bool = False,
    ) -> Generator[ResponseStreamEvent, None, None]:
        """
        Run the agent with streaming responses.
        Args:
            user_input (str): The user's input text.
            prompt_template (Optional[PromptTemplate]): Optional template for history.
            include_internal_events (bool): Whether to emit raw internal events. Defaults to False.
        Yields:
            ResponseStreamEvent: Events representing the agent's progress and output.
        Raises:
            ValueError: If user_input is empty.
            RuntimeError: If the agent exceeds max iterations.
        """
        if not user_input:
            raise ValueError("user_input cannot be empty")
        prompt = self._build_initial_prompt(user_input, prompt_template)
        
        for _ in range(self.max_iterations):
            response_stream = self.llm._chat(
                messages=prompt.to_openai_input(),
                stream=True,
                tools=self._convert_to_openai_tools() if self.tools else None,
                tool_choice=self.tool_choice,
                parallel_tool_calls=self.parallel_tool_calls,
            )
            tool_calls: Dict[str, Dict[str, Any]] = {}

            for event in response_stream:
                if include_internal_events:
                    yield ResponseStreamEvent(
                        type=StreamEventType.INTERNAL,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        raw_event=event,
                    )
                
                if event.type == "response.created":
                    response = event.response
                    yield ResponseStreamEvent(
                        type=StreamEventType.LIFECYCLE,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        run_status=RunStatus.STARTED,
                        stream_status=StreamStatus.IDLE,
                        raw_event=event,
                    )
                    
                elif event.type == "response.output_item.added":
                    item = event.item
                    # Tool call created
                    if item.type == "function_call":
                        tool_calls[item.id] = {
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": "",
                        }
                        yield ResponseStreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            phase=EventPhase.NONE,
                            sequence_number=event.sequence_number,
                            item_id=item.id,
                            tool_name=item.name,
                            call_id=item.call_id,
                            run_status=RunStatus.IN_PROGRESS,
                            stream_status=StreamStatus.STARTED,
                            raw_event=event,
                        )
                    
                    # Message stream created
                    elif item.type == "message":
                        yield ResponseStreamEvent(
                            type=StreamEventType.TEXT,
                            phase=EventPhase.NONE,
                            sequence_number=event.sequence_number,
                            item_id=item.id,
                            run_status=RunStatus.IN_PROGRESS,
                            stream_status=StreamStatus.STARTED,
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
                            sequence_number=event.sequence_number,
                            item_id=event.item_id,
                            tool_name=call["name"],
                            call_id=call["call_id"],
                            arguments=event.delta,
                            run_status=RunStatus.IN_PROGRESS,
                            stream_status=StreamStatus.STREAMING,
                            raw_event=event,
                        )
                        
                # Tool arguments completed
                elif event.type == "response.function_call_arguments.done":
                    call = tool_calls.get(event.item_id)
                    if call:
                        call["arguments"] = event.arguments
                        yield ResponseStreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            phase=EventPhase.FINAL,
                            sequence_number=event.sequence_number,
                            item_id=event.item_id,
                            tool_name=call["name"],
                            call_id=call["call_id"],
                            arguments=event.arguments,
                            run_status=RunStatus.IN_PROGRESS,
                            stream_status=StreamStatus.COMPLETED,
                            raw_event=event,
                        )
                        
                # Text streaming
                elif event.type == "response.output_text.delta":
                    yield ResponseStreamEvent(
                        type=StreamEventType.TEXT,
                        phase=EventPhase.DELTA,
                        sequence_number=event.sequence_number,
                        item_id=event.item_id,
                        text=event.delta,
                        run_status=RunStatus.IN_PROGRESS,
                        stream_status=StreamStatus.STREAMING,
                        raw_event=event,
                    )
                
                # Text completed
                elif event.type == "response.output_text.done":
                    yield ResponseStreamEvent(
                        type=StreamEventType.TEXT,
                        phase=EventPhase.FINAL,
                        sequence_number=event.sequence_number,
                        item_id=event.item_id,
                        text=event.text,
                        run_status=RunStatus.IN_PROGRESS,
                        stream_status=StreamStatus.COMPLETED,
                        raw_event=event,
                    )
                        
                # Response completed successfully
                elif event.type == "response.completed":
                    response = event.response
                    yield ResponseStreamEvent(
                        type=StreamEventType.LIFECYCLE,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        usage=extract_usage_dict(response),
                        run_status=RunStatus.COMPLETED,
                        stream_status=StreamStatus.COMPLETED,
                        raw_event=event,
                    )
                
                # Response incomplete
                elif event.type == "response.incomplete":
                    response = event.response
                    yield ResponseStreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        response_id=response.id,
                        error=response.incomplete_details,  # {'reason': ...}
                        run_status=RunStatus.IN_PROGRESS,
                        stream_status=StreamStatus.INCOMPLETE,
                        raw_event=event,
                    )
                    return
                
                # Response failed (doesn't have sequence_number)
                elif event.type == "response.failed":
                    response = event.response
                    yield ResponseStreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        response_id=response.id,
                        error=response.error,  # {'code': ..., 'message': ...}
                        run_status=RunStatus.FAILED,
                        stream_status=StreamStatus.FAILED,
                        raw_event=event,
                    )
                    return
                
                # Error (doesn't have response_id)
                elif event.type == "error":
                    yield ResponseStreamEvent(
                        type=StreamEventType.ERROR,
                        phase=EventPhase.NONE,
                        sequence_number=event.sequence_number,
                        error={"code": event.code, "message": event.message},  # {'code': ..., 'message': ...}
                        run_status=RunStatus.FAILED,
                        stream_status=StreamStatus.FAILED,
                        raw_event=event,
                    )
                    return
            
            # No tool calls -> agent is done
            if not tool_calls:
                return
            
            # Execute tools and append results to prompt messages
            for tool in tool_calls.values():
                prompt.tool_call(
                    name=tool["name"],
                    arguments=tool["arguments"],
                    call_id=tool["call_id"],
                )
                result = self._execute_tool(
                    tool["name"],
                    json.loads(tool["arguments"]),
                )
                prompt.tool_output(
                    call_id=tool["call_id"],
                    output=str(result),
                )

        raise RuntimeError("Streaming agent exceeded max iterations")
    