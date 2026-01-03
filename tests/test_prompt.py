import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.openai_agent import PromptTemplate, PromptMessage, Role, ContentType


class TestPrompt(unittest.TestCase):
    def test_prompt_message_validation(self):
        # Valid message
        msg = PromptMessage(
            role=Role.USER,
            content_type=ContentType.INPUT_TEXT,
            text="Hello"
        )
        self.assertEqual(msg.to_openai_message()["role"], "user")

        # Invalid: Missing role for text
        with self.assertRaises(ValueError):
            PromptMessage(
                content_type=ContentType.INPUT_TEXT,
                text="Hello"
            ).to_openai_message()

        # Invalid: Missing text for text message
        with self.assertRaises(ValueError):
            PromptMessage(
                role=Role.USER,
                content_type=ContentType.INPUT_TEXT
            ).to_openai_message()

    def test_prompt_template_builder(self):
        template = PromptTemplate()
        template.system("System prompt")
        template.user("User prompt")
        
        messages = template.to_openai_input()
        self.assertEqual(len(messages), 2)
        
        # Check System (Developer) message
        self.assertEqual(messages[0]["role"], "developer")
        self.assertEqual(messages[0]["content"][0]["text"], "System prompt")
        
        # Check User message
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][0]["text"], "User prompt")

    def test_tool_messages(self):
        template = PromptTemplate()
        template.tool_call(name="test_tool", arguments="{}", call_id="123")
        template.tool_output(call_id="123", output="result")
        
        messages = template.to_openai_input()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["type"], "function_call")
        self.assertEqual(messages[1]["type"], "function_call_output")

if __name__ == "__main__":
    unittest.main()
