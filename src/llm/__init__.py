import logging
from src.llm.qwen import BaseLLM, QwenLLM
from src.llm.gemini import GeminiLLM

logger = logging.getLogger(__name__)


def create_llm_engine(config) -> BaseLLM:
    """
    Factory function to instantiate configured LLM provider.

    :param config: PipelineConfig instance
    :return: Concrete BaseLLM instance (QwenLLM or GeminiLLM)
    """
    provider = getattr(config, "llm_provider", "ollama").lower()
    if provider == "gemini":
        logger.info("[LLM] Initializing Gemini LLM provider (model=%s)...", config.gemini_model)
        return GeminiLLM(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
        )
    else:
        logger.info("[LLM] Initializing Ollama/Qwen LLM provider (model=%s)...", config.ollama_model)
        return QwenLLM(
            base_url=config.ollama_base_url,
            model_name=config.ollama_model,
        )


__all__ = ["BaseLLM", "QwenLLM", "GeminiLLM", "create_llm_engine"]
