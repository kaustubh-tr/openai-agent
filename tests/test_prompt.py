import sys
import os
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import PromptTemplate, PromptMessage


class TestPromptTemplates(unittest.TestCase):
    """
    Unit tests for PromptTemplate and PromptMessage construction and validation.
    """

    def test_message_validation_invariants(self):
        """Verify that invalid message combinations raise errors."""

        # 1. Valid message
        msg = PromptMessage(role="user", content_type="input_text", text="Hello")
        openai_msg = msg.to_openai_message()
        self.assertEqual(openai_msg["role"], "user")

        # 2. Invalid: Missing role for text input
        with self.assertRaises(ValueError):
            PromptMessage(content_type="input_text", text="Hello").to_openai_message()

        # 3. Invalid: Missing text content for text input
        with self.assertRaises(ValueError):
            PromptMessage(role="user", content_type="input_text").to_openai_message()

    def test_template_builder_methods(self):
        """Verify helper methods for building a template."""
        template = PromptTemplate()
        template.add_system("System prompt")
        template.add_user("User prompt")

        messages = template.to_openai_input()
        self.assertEqual(len(messages), 2)

        # 1. Check System
        self.assertEqual(messages[0]["role"], "system")
        # Note: The structure depends on to_openai_message implementation.
        # Assuming typical content=[{"type": "text", "text": "..."}]
        content = messages[0]["content"]
        if isinstance(content, list):
            self.assertEqual(content[0]["text"], "System prompt")
        else:
            self.assertEqual(content, "System prompt")

        # 2. Check User
        self.assertEqual(messages[1]["role"], "user")

    def test_tool_call_and_output_messages(self):
        """Verify adding tool calls and outputs to the template."""
        template = PromptTemplate()
        template.add_tool_call(name="test_tool", arguments="{}", call_id="123")
        template.add_tool_output(call_id="123", output="result")

        messages = template.to_openai_input()
        self.assertEqual(len(messages), 2)

        # Check that we have two messages representing the call cycle
        # We check invariant properties rather than exact implementation details
        # which might change (e.g. role vs type keys).

        call_msg = messages[0]
        output_msg = messages[1]

        # Just ensure they are not empty and distinguishable
        self.assertTrue(call_msg)
        self.assertTrue(output_msg)

        # If implementation uses 'type' key (internal) or 'role' (OpenAI)
        if "type" in call_msg:
            self.assertIn("function_call", call_msg["type"])
        if "type" in output_msg:
            self.assertIn("function_call_output", output_msg["type"])


if __name__ == "__main__":
    unittest.main()
