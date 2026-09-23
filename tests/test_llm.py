"""
Unit tests for Qwen LLM module.
"""

import unittest
from src.llm.qwen import QwenLLM


class TestQwenLLM(unittest.TestCase):
    """Test Qwen LLM Ollama interface."""

    def test_llm_init(self) -> None:
        llm = QwenLLM(base_url="http://localhost:11434", model_name="qwen2.5")
        self.assertEqual(llm.model_name, "qwen2.5")
        self.assertTrue(isinstance(llm.is_available, bool))

    def test_empty_prompt(self) -> None:
        llm = QwenLLM()
        response = llm.generate_response("")
        self.assertEqual(response, "")


from unittest.mock import MagicMock, patch
from src.llm.gemini import GeminiLLM


class TestGeminiLLM(unittest.TestCase):
    """Test Gemini LLM REST interface."""

    def test_gemini_init(self) -> None:
        llm = GeminiLLM(api_key="test_key", model_name="gemini-3-flash-preview")
        self.assertEqual(llm.model_name, "gemini-3-flash-preview")

    def test_empty_prompt(self) -> None:
        llm = GeminiLLM(api_key="test_key")
        self.assertEqual(llm.generate_response(""), "")

    @patch("requests.Session.post")
    def test_valid_response(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "AI is artificial intelligence."}]
                    },
                    "finishReason": "STOP"
                }
            ]
        }
        mock_post.return_value = mock_resp

        llm = GeminiLLM(api_key="test_key")
        response = llm.generate_response("Tell me something about AI")
        self.assertEqual(response, "AI is artificial intelligence.")

    @patch("requests.Session.post")
    def test_empty_or_blocked_response(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": []},
                    "finishReason": "MAX_TOKENS"
                }
            ],
            "promptFeedback": {"blockReason": "SAFETY"}
        }
        mock_post.return_value = mock_resp

        llm = GeminiLLM(api_key="test_key")
        response = llm.generate_response("Test prompt")
        self.assertIn("I am sorry", response)
        self.assertNotIn("[Gemini Error: Received empty response from API]", response)

    @patch("requests.Session.post")
    def test_rate_limit_429_response(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"error": {"code": 429, "message": "Resource exhausted"}}'
        mock_post.return_value = mock_resp

        llm = GeminiLLM(api_key="test_key")
        response = llm.generate_response("Rate limit test prompt")
        self.assertIn("too many requests", response)
        self.assertNotIn("[Gemini Error: HTTP 429]", response)


if __name__ == "__main__":
    unittest.main()
