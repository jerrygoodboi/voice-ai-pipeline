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
    "Keep responses natural, friendly, and conversational. "
    "Keep general responses concise unless the user specifically asks for a story, detailed explanation, or lengthier output. "
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
        model_name: str | None = None,
        timeout: float = 6.0,
    ) -> None:
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
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
            logger.info("[LLM] Initialized Gemini client (model=%s, IPv4 optimized, timeout=%.1fs).", self.model_name, self.timeout)

    def _call_model(
        self,
        model: str,
        prompt: str,
        timeout: float,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str | None, int]:
        """Helper to invoke a specific Gemini model with low latency settings."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={self.api_key}"
        )
        gen_config: dict[str, Any] = {
            "temperature": 0.5,
            "maxOutputTokens": 500,
        }
        if "gemini-3" in model.lower() or "2.5" in model.lower():
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        contents: list[dict[str, Any]] = []
        if history:
            for turn in history:
                role = "user" if turn.get("role") == "user" else "model"
                text = turn.get("content", "").strip()
                if text:
                    contents.append({"role": role, "parts": [{"text": text}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "generationConfig": gen_config,
        }

        resp = self.session.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    txt = parts[0].get("text", "").strip()
                    if txt:
                        return txt, 200
            return None, 200
        elif resp.status_code in (429, 503):
            logger.warning("[LLM] Model '%s' HTTP %d (rate limit/demand spike).", model, resp.status_code)
            return None, resp.status_code
        else:
            logger.warning("[LLM] Model '%s' HTTP %d: %s", model, resp.status_code, resp.text[:120])
            return None, resp.status_code

    def generate_response(
        self, prompt: str, history: list[dict[str, str]] | None = None
    ) -> str:
        """
        Send text prompt to Gemini API and return concise conversational text response.
        Uses fast failover across high-speed models to prevent any hanging.
        """
        if not prompt or not prompt.strip():
            return ""

        if not self.api_key:
            error_msg = "[Gemini Error: GEMINI_API_KEY is missing or empty]"
            logger.error("[LLM] %s", error_msg)
            return error_msg

        logger.info("[LLM] Sending text to Gemini (%s)...", self.model_name)

        models_to_try = [self.model_name]
        for fallback in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        saw_429 = False
        saw_empty_200 = False

        for m in models_to_try:
            try:
                ans, status_code = self._call_model(m, prompt, timeout=self.timeout, history=history)
                if ans:
                    logger.info("[LLM] GEMINI RESPONSE (%s): '%s'", m, ans)
                    return ans
                if status_code == 429:
                    saw_429 = True
                elif status_code == 200:
                    saw_empty_200 = True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as net_err:
                logger.warning("[LLM] Model '%s' timed out or network error (%s). Trying fallback...", m, net_err)
                continue
            except Exception as e:
                logger.error("[LLM] Model '%s' error: %s. Trying fallback...", m, e)
                continue

        if saw_429:
            return "I am currently receiving too many requests. Please try again in a few seconds."
        if saw_empty_200:
            return "I am sorry, I couldn't generate a response for that."
        return "I am sorry, I couldn't reach the voice assistant server in time. Please try again."
