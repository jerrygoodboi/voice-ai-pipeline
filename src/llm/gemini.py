"""
Language Model (LLM) component using Google's Gemini API.
Optimized for real-time voice: forces IPv4 to eliminate Linux IPv6 routing timeouts,
disables chain-of-thought thinking delay, and keeps responses concise.
"""

import logging
import os
import socket
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.connection import allowed_gai_family

from src.llm.qwen import BaseLLM

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a helpful, fast, and natural voice assistant. "
    "Keep responses concise (1 to 2 sentences maximum), friendly, and conversational. "
    "Do NOT use markdown formatting, bullet points, asterisks, or emoji, as your response will be read aloud by a text-to-speech engine."
)


class IPv4HTTPAdapter(HTTPAdapter):
    """Force IPv4 connection to prevent Linux IPv6 DNS/routing timeouts."""

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        kwargs["socket_options"] = kwargs.get("socket_options", [])
        super().init_poolmanager(*args, **kwargs)


class GeminiLLM(BaseLLM):
    """
    Gemini LLM client communicating with Google Gemini API.
    Uses an optimized IPv4 session with direct REST API to minimize latency for voice conversations.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-flash-lite-latest",
        timeout: float = 15.0,
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("GEMINI_API_KEY", "")

        # Build optimized session forcing IPv4
        self.session = requests.Session()
        adapter = IPv4HTTPAdapter()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Force urllib3 to only resolve IPv4
        try:
            import requests.packages.urllib3.util.connection as urllib3_cn
            urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
        except Exception:
            pass

        if not self.api_key:
            logger.warning(
                "[LLM] GEMINI_API_KEY is not set. Gemini API calls will fail until a valid key is provided."
            )
        else:
            logger.info("[LLM] Initialized Gemini client (model=%s, IPv4 optimized).", self.model_name)

    def generate_response(self, prompt: str) -> str:
        """
        Send text prompt to Gemini API and return conversational text response.
        """
        if not prompt or not prompt.strip():
            return ""

        if not self.api_key:
            error_msg = "[Gemini Error: GEMINI_API_KEY is missing or empty]"
            logger.error("[LLM] %s", error_msg)
            return error_msg

        logger.info("[LLM] Sending text to Gemini (%s)...", self.model_name)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            f"?key={self.api_key}"
        )

        gen_config = {
            "temperature": 0.7,
            "maxOutputTokens": 300,
        }
        if "gemini-3" in self.model_name.lower():
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "generationConfig": gen_config,
        }

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    prompt_feedback = data.get("promptFeedback", {})
                    block_reason = prompt_feedback.get("blockReason")
                    finish_reason = candidates[0].get("finishReason") if candidates else None

                    text = ""
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()

                    if text:
                        logger.info("[LLM] GEMINI RESPONSE: '%s'", text)
                        return text

                    logger.warning(
                        "[LLM] Gemini returned an empty response. Model: %s | Candidate count: %d | Finish reason: %s | Block reason: %s | Response: %s",
                        self.model_name,
                        len(candidates),
                        finish_reason,
                        block_reason,
                        data,
                    )
                    return "I am sorry, I couldn't generate a response for that."
                elif response.status_code in (429, 503):
                    logger.warning(
                        "[LLM] Model '%s' HTTP %d (rate limit/high demand). Retrying with backoff and fallback model...",
                        self.model_name,
                        response.status_code,
                    )
                    import time
                    time.sleep(1.5)

                    fallback_url = (
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent"
                        f"?key={self.api_key}"
                    )
                    fb_payload = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300},
                    }
                    fb_resp = self.session.post(fallback_url, json=fb_payload, timeout=self.timeout)
                    if fb_resp.status_code == 200:
                        data = fb_resp.json()
                        candidates = data.get("candidates", [])
                        prompt_feedback = data.get("promptFeedback", {})
                        block_reason = prompt_feedback.get("blockReason")
                        finish_reason = candidates[0].get("finishReason") if candidates else None

                        text = ""
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            if parts:
                                text = parts[0].get("text", "").strip()

                        if text:
                            logger.info("[LLM] GEMINI RESPONSE: '%s'", text)
                            return text

                        logger.warning(
                            "[LLM] Gemini fallback returned empty text response. Model: gemini-flash-lite-latest | Candidate count: %d | Finish reason: %s | Block reason: %s",
                            len(candidates),
                            finish_reason,
                            block_reason,
                        )
                    else:
                        logger.error(
                            "[LLM] Gemini fallback failed [%d]: %s",
                            fb_resp.status_code,
                            fb_resp.text,
                        )

                    if response.status_code == 429 or fb_resp.status_code == 429:
                        return "I am currently receiving too many requests. Please try again in a few seconds."
                    return "I am sorry, the AI service is currently unavailable. Please try again shortly."
                else:
                    logger.error(
                        "[LLM] Gemini request failed [%d]: %s",
                        response.status_code,
                        response.text,
                    )
                    return "I am sorry, the AI service encountered an error."

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                if attempt < max_attempts:
                    logger.warning("[LLM] Network transient issue (%s). Retrying in 1s (attempt %d/%d)...", net_err, attempt, max_attempts)
                    import time
                    time.sleep(1.0)
                    continue
                logger.error("[LLM] Gemini connection failed after retries: %s", net_err)
                return "[Gemini Error: Network connection error]"
            except Exception as e:
                error_msg = f"[Gemini API Error: {e}]"
                logger.error("[LLM] %s", error_msg)
                return error_msg
