import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI


def main():
    # 1. Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # 2. Prepare messages
    messages = [
        {"role": "system", "content": "You are a poetic assistant."},
        {"role": "user", "content": "Write a haiku about recursion."},
    ]

    print("User: Write a haiku about recursion.")
    print("Assistant: ", end="", flush=True)

    # 3. Call stream
    try:
        # The stream method returns a generator of OpenAI response events
        stream = llm.stream(messages=messages)

        for event in stream:
            # Only print text delta events for clean, readable streaming output
            if getattr(event, "type", None) == "response.output_text.delta":
                print(getattr(event, "delta", ""), end="", flush=True)
        print()  # Final newline after streaming is complete

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
