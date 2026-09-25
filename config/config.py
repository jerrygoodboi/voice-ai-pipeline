"""
Configuration module for the voice AI pipeline.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass
class PipelineConfig:
    """Configuration settings for the Voice AI Pipeline components."""

    # VAD (Silero) Settings
    vad_device: str = os.getenv("VAD_DEVICE", "cpu")
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.7"))
    vad_min_speech_duration: float = float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.4"))
    vad_min_silence_duration: float = float(os.getenv("VAD_MIN_SILENCE_DURATION", "0.5"))

    # STT (Whisper) Settings
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # LLM Settings (Provider: "gemini" or "ollama")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")

    # Gemini Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

    # Ollama / Qwen Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

    # TTS Settings (Engine: "edge", "piper", or "kokoro")
    tts_engine: str = os.getenv("TTS_ENGINE", "edge")

    # Edge TTS Settings
    edge_voice: str = os.getenv("EDGE_VOICE", "en-US-JennyNeural")

    # Piper TTS Settings
    piper_voice: str = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
    piper_model_path: str = os.getenv("PIPER_MODEL_PATH", "")

    # Kokoro TTS Settings
    kokoro_voice: str = os.getenv("KOKORO_VOICE", "af_heart")
    kokoro_device: str = os.getenv("KOKORO_DEVICE", "cpu")

    # Audio Settings
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # Pipeline Mode: "standalone" (all local), "client" (mic/vad on laptop), "server" (remote processing)
    pipeline_mode: str = os.getenv("PIPELINE_MODE", "standalone")

    # Remote Server / Tailscale Settings
    remote_server_url: str = os.getenv("REMOTE_SERVER_URL", "http://localhost:8000")
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))
    remote_request_timeout: float = float(os.getenv("REMOTE_REQUEST_TIMEOUT", "60.0"))


def get_config() -> PipelineConfig:
    """Helper function to return a fresh PipelineConfig instance."""
    return PipelineConfig()
