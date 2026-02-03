import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
        return
    
    # 1. Configure your llm
    llm = ChatOpenAI(
        model="gpt-4.1-mini", 
        temperature=0.7,
        timeout=10,
        max_output_tokens=1000,
    )

    # 2. Define the messages with chat history
    user_input = "Tell me a joke."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
        {"role": "user", "content": user_input},
    ]
    
    print("\n\n--- Non-Streaming LLM Response ---\n\n")
    
    # 3. Run llm without streaming
    print(f"User: {user_input}")
    result = llm.invoke(messages=messages)
    # print(f"Assistant: \n{result}")  # for complete response by OpenAI
    print(f"Assistant: {result.output_text}")
        
    print("\n\n--- Streaming LLM Response ---\n\n")
    
    # 4. Run llm with streaming
    print("Assistant: ", end="", flush=True)
    
    for event in llm.stream(messages):
        # print(event)  # for complete list of events from OpenAI
        if getattr(event, "type", None) == "response.output_text.delta":
            delta_text = getattr(event, "delta", "")
            if isinstance(delta_text, str):
                print(delta_text, end="", flush=True)


if __name__ == "__main__":
    main()
