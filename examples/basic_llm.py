import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI


def main():
    # 1. Initialize the LLM directly
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # 2. Prepare messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]

    # 3. Call chat (using invoke)
    print("Sending request...")
    try:
        response = llm.invoke(messages)
        # result is the raw OpenAI Responses API response object
        print(f"Result: {response}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
