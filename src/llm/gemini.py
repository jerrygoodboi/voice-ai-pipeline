"""
Language Model (LLM) component using Google's Gemini API.
Uses Google's official google-genai SDK.
"""

import logging
import os
from typing import Any
from src.llm.qwen import BaseLLM

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import errors
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GeminiLLM(BaseLLM):
    """
    Gemini LLM client using Google GenAI SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY", "") if api_key is None else api_key

        self.client = client
        if self.client is None and GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("[LLM] Initialized Gemini client with model '%s'.", self.model_name)
            except Exception as e:
                logger.error("[LLM] Failed to initialize Google GenAI client: %s", e)
                self.client = None

        if not self.api_key:
            logger.warning(
                "[LLM] GEMINI_API_KEY is not set. Gemini API calls will fail until a valid key is provided."
            )

    def generate_response(self, prompt: str) -> str:
        """
        Send text prompt to Gemini API and return text response.
        """
        if not prompt or not prompt.strip():
            return ""

        if not self.api_key:
            error_msg = "[Gemini Error: GEMINI_API_KEY environment variable is missing or empty]"
            logger.error("[LLM] %s", error_msg)
            return error_msg

        if not GENAI_AVAILABLE:
            error_msg = "[Gemini Error: 'google-genai' package is not installed]"
            logger.error("[LLM] %s", error_msg)
            return error_msg

        if self.client is None:
            # Attempt lazy initialization if client was not built
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                error_msg = f"[Gemini Initialization Error: {e}]"
                logger.error("[LLM] %s", error_msg)
                return error_msg

        logger.info("[LLM] Sending text to Gemini (%s)...", self.model_name)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text_response = (response.text or "").strip() if hasattr(response, "text") else ""
            if not text_response:
                logger.warning("[LLM] Received empty response from Gemini API.")
                return "[Gemini Error: Received empty response from API]"

            logger.info("[LLM] Gemini response received: %s", text_response)
            return text_response

        except Exception as e:
            error_msg = f"[Gemini API Error: {e}]"
            logger.error("[LLM] %s", error_msg)
            return error_msg
