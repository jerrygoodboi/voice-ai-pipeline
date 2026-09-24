"""
Language Model (LLM) component using local Ollama REST API for Qwen.
Communicates with local Ollama service endpoint.
"""

from abc import ABC, abstractmethod
import logging
import requests

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract interface for Large Language Models."""

    @abstractmethod
    def generate_response(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """
        Generate text response for a given prompt string with optional conversation history.

        :param prompt: User prompt text
        :param history: Optional list of previous turns [{'role': 'user'|'model', 'content': str}]
        :return: Generated text response
        """
        pass


class QwenLLM(BaseLLM):
    """
    Qwen LLM client communicating with Ollama REST API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5-coder:7b",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

        self.is_available: bool = self.check_ollama_status()

    def check_ollama_status(self) -> bool:
        """
        Check if Ollama service is reachable and log status.
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=3.0)
            if response.status_code == 200:
                models_info = response.json().get("models", [])
                installed_names = [m.get("name", "") for m in models_info]
                installed_models = [name.split(":")[0] for name in installed_names]
                logger.info(
                    "[LLM] Connected to Ollama at %s. Installed models: %s",
                    self.base_url,
                    installed_names,
                )

                # Verify configured model is present (exact match, base name match, or tag prefix)
                model_base = self.model_name.split(":")[0]
                is_present = (
                    self.model_name in installed_names
                    or any(model_base == m for m in installed_models)
                    or any(name.startswith(self.model_name) or self.model_name.startswith(name) for name in installed_names)
                )
                if not is_present:
                    logger.warning(
                        "[LLM] Model '%s' was not found in local Ollama instance. Run `ollama pull %s` in terminal.",
                        self.model_name,
                        self.model_name,
                    )
                return True
            else:
                logger.warning(
                    "[LLM] Ollama returned status code %d at %s.",
                    response.status_code,
                    self.base_url,
                )
                return False
        except requests.exceptions.ConnectionError:
            logger.warning(
                "[LLM] Could not connect to Ollama service at %s. Please ensure Ollama is installed and running (`ollama serve`).",
                self.base_url,
            )
            return False
        except Exception as e:
            logger.error("[LLM] Ollama health check error: %s", e)
            return False

    def generate_response(self, prompt: str, history: list[dict[str, str]] | None = None) -> str:
        """
        Send text prompt to Qwen via Ollama REST API and return text response.
        """
        if not prompt or not prompt.strip():
            return ""

        logger.info("[LLM] Sending text to Qwen...")

        full_prompt = prompt
        if history:
            context_lines = []
            for turn in history:
                role = "User" if turn.get("role") == "user" else "Assistant"
                context_lines.append(f"{role}: {turn.get('content', '')}")
            context_lines.append(f"User: {prompt}")
            context_lines.append("Assistant:")
            full_prompt = "\n".join(context_lines)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "keep_alive": -1,  # Pin model in VRAM indefinitely (no cold-load lag)
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                text_response = data.get("response", "").strip()
                logger.info("[LLM] Response received: %s", text_response)
                return text_response
            else:
                logger.error(
                    "[LLM] Ollama request failed with status code %d: %s",
                    response.status_code,
                    response.text,
                )
                return f"[Ollama Error: HTTP {response.status_code}]"
        except requests.exceptions.ConnectionError:
            error_msg = f"[Ollama Service Offline] Could not reach {self.base_url}. Run `ollama serve` and `ollama pull {self.model_name}`."
            logger.error("[LLM] %s", error_msg)
            return error_msg
        except Exception as e:
            logger.error("[LLM] Error generating Qwen response: %s", e)
            return f"[LLM Error: {e}]"
