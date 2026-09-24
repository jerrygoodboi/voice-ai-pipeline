"""
Speech-to-Text (STT) component using faster-whisper.
Transcribes audio speech segments into text strings.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger(__name__)


class BaseSTT(ABC):
    """Abstract interface for Speech-to-Text engines."""

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray | bytes) -> str:
        """
        Transcribe audio into text.

        :param audio_data: Audio samples or raw bytes
        :return: Transcribed text string
        """
        pass


class WhisperSTT(BaseSTT):
    """
    Whisper Speech-to-Text implementation using faster-whisper.
    Runs on CPU or CUDA hardware depending on configuration.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None

        self._init_model()

    def _init_model(self) -> None:
        """Initialize faster-whisper model."""
        if WhisperModel is None:
            logger.error("[STT] faster-whisper package is not installed.")
            return

        try:
            logger.info(
                "[STT] Loading Whisper model '%s' (device=%s, compute_type=%s)...",
                self.model_name,
                self.device,
                self.compute_type,
            )
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("[STT] Whisper model '%s' loaded successfully.", self.model_name)
        except Exception as e:
            logger.error("[STT] Failed to load Whisper model '%s': %s", self.model_name, e)
            logger.info("[STT] Attempting fallback with cpu / float32...")
            try:
                self.model = WhisperModel(self.model_name, device="cpu", compute_type="float32")
                logger.info("[STT] Fallback Whisper model loaded on CPU.")
            except Exception as fallback_err:
                logger.critical("[STT] Whisper model loading failed completely: %s", fallback_err)
                self.model = None

    def transcribe(self, audio_data: np.ndarray | bytes, beam_size: int = 1) -> str:
        """
        Transcribe speech audio array into text string.
        Ignores empty or silent audio inputs.
        """
        if audio_data is None or len(audio_data) == 0:
            logger.debug("[STT] Empty audio data provided for transcription.")
            return ""

        if self.model is None:
            logger.error("[STT] Whisper model is not loaded. Cannot transcribe.")
            return ""

        try:
            logger.info("[STT] Transcribing speech...")
            
            # Ensure float32 1D numpy array normalized to [-1.0, 1.0]
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data.astype(np.float32)

            segments, info = self.model.transcribe(
                audio_array,
                beam_size=beam_size,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400, threshold=0.4),
            )

            text_chunks = [segment.text.strip() for segment in segments if segment.text.strip()]
            full_text = " ".join(text_chunks).strip()

            if full_text:
                logger.info("[STT] Text: %s", full_text)
            else:
                logger.info("[STT] No audible text transcribed.")

            return full_text
        except Exception as e:
            logger.error("[STT] Transcription error: %s", e)
            return ""
