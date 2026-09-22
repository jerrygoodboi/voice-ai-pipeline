"""
Unit tests for Gemini LLM component and provider selection.
"""

import unittest
from unittest.mock import MagicMock, patch
from config.config import PipelineConfig
from src.llm import create_llm_engine, QwenLLM, GeminiLLM


class TestGeminiLLM(unittest.TestCase):
    """Test Gemini LLM client and error handling."""

    def test_gemini_init_with_key(self) -> None:
        """Test Gemini initialization with explicit API key."""
        mock_client = MagicMock()
        llm = GeminiLLM(api_key="test_api_key", model_name="gemini-2.5-flash", client=mock_client)
        self.assertEqual(llm.model_name, "gemini-2.5-flash")
        self.assertEqual(llm.api_key, "test_api_key")
        self.assertEqual(llm.client, mock_client)

    def test_gemini_init_without_key(self) -> None:
        """Test Gemini initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            llm = GeminiLLM(api_key="", model_name="gemini-2.5-flash")
            self.assertEqual(llm.api_key, "")

    def test_empty_prompt(self) -> None:
        """Test empty prompt returns empty string."""
        mock_client = MagicMock()
        llm = GeminiLLM(api_key="test_key", client=mock_client)
        response = llm.generate_response("")
        self.assertEqual(response, "")
        mock_client.models.generate_content.assert_not_called()

    def test_missing_api_key_response(self) -> None:
        """Test missing API key generates explicit error message."""
        llm = GeminiLLM(api_key="")
        response = llm.generate_response("Hello")
        self.assertIn("Gemini Error", response)
        self.assertIn("GEMINI_API_KEY", response)

    def test_successful_generate_response(self) -> None:
        """Test mocked successful Gemini API response generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello! How can I help you today?"
        mock_client.models.generate_content.return_value = mock_response

        llm = GeminiLLM(api_key="valid_key", model_name="gemini-2.5-flash", client=mock_client)
        response = llm.generate_response("Hi Gemini")

        self.assertEqual(response, "Hello! How can I help you today?")
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-2.5-flash",
            contents="Hi Gemini",
        )

    def test_empty_gemini_api_response(self) -> None:
        """Test handling of empty response from Gemini API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "  "
        mock_client.models.generate_content.return_value = mock_response

        llm = GeminiLLM(api_key="valid_key", client=mock_client)
        response = llm.generate_response("Testing empty")

        self.assertIn("Gemini Error", response)
        self.assertIn("empty response", response.lower())

    def test_gemini_api_exception_handling(self) -> None:
        """Test handling of Gemini API exceptions."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API connection timeout")

        llm = GeminiLLM(api_key="valid_key", client=mock_client)
        response = llm.generate_response("Hello")

        self.assertIn("Gemini API Error", response)
        self.assertIn("API connection timeout", response)


class TestLLMProviderSelection(unittest.TestCase):
    """Test LLM factory provider selection."""

    def test_default_provider_selection(self) -> None:
        """Test default LLM provider is Ollama/Qwen."""
        cfg = PipelineConfig(llm_provider="ollama")
        llm = create_llm_engine(cfg)
        self.assertIsInstance(llm, QwenLLM)

    def test_gemini_provider_selection(self) -> None:
        """Test selecting Gemini LLM provider."""
        cfg = PipelineConfig(
            llm_provider="gemini",
            gemini_api_key="test_gemini_key",
            gemini_model="gemini-2.5-flash",
        )
        llm = create_llm_engine(cfg)
        self.assertIsInstance(llm, GeminiLLM)
        self.assertEqual(llm.model_name, "gemini-2.5-flash")
        self.assertEqual(llm.api_key, "test_gemini_key")


if __name__ == "__main__":
    unittest.main()
