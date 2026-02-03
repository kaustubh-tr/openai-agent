import asyncio
import os
import sys

# Ensure the package is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from literun import ChatOpenAI


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable.")
        return

    # 1. Configure your llm
    # It supports both sync (invoke/stream) and async (ainvoke/astream) methods
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.7,
        timeout=10,
        max_output_tokens=1000,
    )

    # 2. Run llm asynchronously without streaming
    print("\n--- 1. Simple Async Invocation ---")
    messages = [{"role": "user", "content": "Tell me a haiku about recursion."}]
    response = await llm.ainvoke(messages)

    print(f"Response:\n{response.output_text}")

    # 3. Run llm asynchronously with streaming
    print("\n--- 2. Async Streaming ---")
    stream_messages = [{"role": "user", "content": "Count from 1 to 5 slowly."}]
    print("Streaming Response: ", end="", flush=True)

    async for event in llm.astream(messages=stream_messages):
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
