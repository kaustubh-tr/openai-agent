from typing import List, Dict, Any

def extract_output_text(response: Any) -> str:
    """
    Extracts and concatenates output text from the response object.
    Args:
        response (Any): The response object containing output messages.
    Returns:
        str: The concatenated output text.
    """
    texts: List[str] = []
    for output in response.output:
        if output.type == "message":
            for content in output.content:
                if content.type == "output_text":
                    texts.append(content.text)

    return "".join(texts)

def extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool call information from the response object into a list of dictionaries.
    Args:
        response (Any): The response object containing tool call information.
    Returns:
        List[Dict[str, Any]]: A list of dictionaries with tool call details.
    """
    import json
    tool_calls: List[Dict[str, Any]] = []
    for output in response.output:
        if output.type == "function_call":
            tool_calls.append({
                "arguments": json.loads(output.arguments),
                "call_id": output.call_id,
                "name": output.name,
                "type": output.type,
                "id": output.id,
                "status": output.status
            })
    return tool_calls

def extract_usage_dict(response: Any) -> Dict[str, Any]:
    """
    Extracts usage statistics from the response object into a dictionary.
    Args:
        response (Any): The response object containing usage statistics.
    Returns:
        Dict[str, Any]: A dictionary with usage statistics.
    """
    return {
        "input_tokens": response.usage.input_tokens,
        "input_tokens_details": {
            "cached_tokens": response.usage.input_tokens_details.cached_tokens
        },
        "output_tokens": response.usage.output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens
        },
        "total_tokens": response.usage.total_tokens
    }
