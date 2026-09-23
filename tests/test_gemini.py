"""
Unit tests for GeminiLLM module.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.llm.gemini import GeminiLLM


class TestGeminiLLM(unittest.TestCase):
    """Test Gemini LLM REST and offline interface."""

    def test_gemini_init(self) -> None:
        llm = GeminiLLM(api_key="test_key", model_name="gemini-flash-lite-latest")
        self.assertEqual(llm.model_name, "gemini-flash-lite-latest")
        self.assertEqual(llm.api_key, "test_key")

    def test_empty_prompt(self) -> None:
        llm = GeminiLLM(api_key="test_key")
        resp = llm.generate_response("")
        self.assertEqual(resp, "")

    def test_missing_api_key(self) -> None:
        llm = GeminiLLM(api_key="")
        resp = llm.generate_response("Hello")
        self.assertIn("missing or empty", resp)

    @patch("src.llm.gemini.requests.Session.post")
    def test_mocked_gemini_response(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! I am ready to help."}]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        llm = GeminiLLM(api_key="mock_key")
        resp = llm.generate_response("Hello")
        self.assertEqual(resp, "Hello! I am ready to help.")


if __name__ == "__main__":
    unittest.main()
