"""
Factory module to instantiate the configured TTS engine.
Supports 'edge', 'piper', and 'kokoro'.
"""

import logging
from config.config import PipelineConfig
from src.tts.kokoro import BaseTTS, KokoroTTS
from src.tts.edge import EdgeTTS
from src.tts.piper import PiperTTS

logger = logging.getLogger(__name__)


def get_tts_engine(config: PipelineConfig) -> BaseTTS:
    """
    Instantiate and return the appropriate TTS engine based on config.tts_engine.

    :param config: PipelineConfig instance
    :return: An instance implementing BaseTTS
    """
    engine_type = (config.tts_engine or "edge").lower().strip()

    if engine_type in ("edge", "edge-tts", "azure"):
        logger.info("[TTS Factory] Selecting EdgeTTS engine (voice=%s)...", config.edge_voice)
        return EdgeTTS(
            voice=config.edge_voice,
            sample_rate=config.sample_rate,
        )

    elif engine_type in ("piper", "piper-tts"):
        logger.info("[TTS Factory] Selecting PiperTTS engine (voice=%s)...", config.piper_voice)
        return PiperTTS(
            voice=config.piper_voice,
            model_path=config.piper_model_path,
            sample_rate=config.sample_rate,
        )

    elif engine_type in ("kokoro", "kokoro-onnx"):
        logger.info("[TTS Factory] Selecting KokoroTTS engine (voice=%s)...", config.kokoro_voice)
        return KokoroTTS(
            voice=config.kokoro_voice,
            sample_rate=config.sample_rate,
            device=config.kokoro_device,
        )

    else:
        logger.warning(
            "[TTS Factory] Unknown TTS engine '%s'. Defaulting to EdgeTTS.",
            engine_type,
        )
        return EdgeTTS(
            voice=config.edge_voice,
            sample_rate=config.sample_rate,
        )
