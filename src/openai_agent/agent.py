from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from .tool import Tool
from .prompt_template import PromptTemplate
from .prompt_message import PromptMessage
from .constants import Role, ContentType


class Agent:
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        parallel_tool_calls: bool = True,
        max_iterations: int = 10,
        **kwargs: Any,
    ):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt
        self.parallel_tool_calls = parallel_tool_calls
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = max_iterations
        self.model_kwargs = kwargs

    # ------------------------
    # Tool registration
    # ------------------------
    def add_tool(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    # ------------------------
    # LLM call
    # ------------------------
    def _chat(self, messages: List[Dict[str, Any]]) -> Any:
        kwargs = {
            "model": self.model,
            "input": messages,
            "parallel_tool_calls": self.parallel_tool_calls,
            **self.model_kwargs,
        }

        if self.tools:
            kwargs["tools"] = [tool.to_openai_tool() for tool in self.tools.values()]
            kwargs["tool_choice"] = "auto"

        return self.client.responses.create(**kwargs)

    # ------------------------
    # Tool execution
    # ------------------------
    def _execute_tool(self, name: str, raw_args: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        parsed_args = tool.resolve_arguments(raw_args)
        return tool.func(**parsed_args)

    # ------------------------
    # Main agent loop
    # ------------------------
    def invoke(
        self,
        user_input: str,
        prompt_template: Optional[PromptTemplate] = None
    ) -> str:
        
        if not user_input:
            raise ValueError("user_input cannot be empty")
        
        messages: List[Dict[str, Any]] = []
        
        # Build initial history
        if prompt_template:
            messages.extend(prompt_template.to_openai_input())
        
        elif self.system_prompt:
            messages.append(
                PromptMessage(
                    role=Role.DEVELOPER,
                    content_type=ContentType.INPUT_TEXT,
                    text=self.system_prompt,
                ).to_openai_message()
            )

        messages.append(
            PromptMessage(
                role=Role.USER,
                content_type=ContentType.INPUT_TEXT,
                text=user_input,
            ).to_openai_message()
        )

        for _ in range(self.max_iterations):
            response = self._chat(messages)

            final_output = None
            tool_called = False

            for item in response.output:
                # Tool call event
                if item.type == ContentType.FUNCTION_CALL:
                    tool_called = True
                    
                    # 1. Append tool call
                    messages.append(
                        PromptMessage(
                            content_type=ContentType.FUNCTION_CALL,
                            call_id=item.call_id,
                            name=item.name,
                            arguments=item.arguments,
                        ).to_openai_message()
                    )

                    # 2. Execute tool
                    raw_args = json.loads(item.arguments)
                    result = self._execute_tool(item.name, raw_args)

                    # 3. Append tool output
                    messages.append(
                        PromptMessage(
                            content_type=ContentType.FUNCTION_CALL_OUTPUT,
                            call_id=item.call_id,
                            output=str(result),
                        ).to_openai_message()
                    )

                # Final assistant message
                elif item.type == "message":
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