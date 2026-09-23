"""
Voice Activity Detection (VAD) component using Silero VAD (ONNX Runtime).
Detects speech start/end boundaries and extracts speech segments.
"""

from abc import ABC, abstractmethod
import logging
import os
import urllib.request
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)

# Official Silero VAD v4 ONNX model URL
SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
SILERO_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "silero_vad")
SILERO_MODEL_PATH = os.path.join(SILERO_MODEL_DIR, "silero_vad.onnx")


class BaseVAD(ABC):
    """Abstract interface for Voice Activity Detection."""

    @abstractmethod
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Determine if audio chunk contains speech."""
        pass

    @abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        """
        Process audio chunk and return accumulated speech segment when speech ends.

        :param audio_chunk: 1D numpy array of audio samples (float32, 16kHz mono)
        :return: 1D numpy array of speech segment when speech ends, otherwise None.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""
        pass


class SileroVAD(BaseVAD):
    """
    Silero VAD implementation using ONNX Runtime.
    Processes continuous audio chunks and returns complete speech segments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        min_silence_duration: float = 0.5,
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.device = device

        self.session = None
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

        # Internal state tracking
        self.is_speaking: bool = False
        self.speech_chunks: list[np.ndarray] = []
        self.speech_duration: float = 0.0
        self.silence_duration: float = 0.0

        self._init_model()

    def _init_model(self) -> None:
        """Load or download Silero VAD ONNX model."""
        if ort is None:
            logger.warning("[VAD] onnxruntime is not installed. Falling back to energy VAD.")
            return

        try:
            if not os.path.exists(SILERO_MODEL_PATH):
                logger.info("[VAD] Downloading Silero VAD ONNX model to %s...", SILERO_MODEL_PATH)
                os.makedirs(SILERO_MODEL_DIR, exist_ok=True)
                urllib.request.urlretrieve(SILERO_VAD_URL, SILERO_MODEL_PATH)
                logger.info("[VAD] Silero VAD ONNX model downloaded successfully.")

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            
            providers = ["CPUExecutionProvider"]
            if self.device.lower() in ("cuda", "gpu"):
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(
                SILERO_MODEL_PATH,
                providers=providers,
                sess_options=options,
            )
            logger.info("[VAD] Silero VAD initialized on device '%s'.", self.device)
        except Exception as e:
            logger.error("[VAD] Failed to initialize Silero VAD ONNX session: %s", e)
            logger.info("[VAD] Will use energy-based VAD fallback.")
            self.session = None

    def reset(self) -> None:
        """Reset VAD internal state and buffers."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self.is_speaking = False
        self.speech_chunks.clear()
        self.speech_duration = 0.0
        self.silence_duration = 0.0

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Calculate speech probability for audio chunk.

        :param audio_chunk: 1D numpy array of float32 samples.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        # If ONNX session loaded, run Silero inference
        if self.session is not None:
            try:
                # Silero VAD v4 expects input shape (1, N) and sample rate tensor
                audio_input = audio_chunk.reshape(1, -1).astype(np.float32)
                sr_input = np.array(self.sample_rate, dtype=np.int64)

                ort_inputs = {
                    "input": audio_input,
                    "sr": sr_input,
                    "h": self._h,
                    "c": self._c,
                }
                out, self._h, self._c = self.session.run(None, ort_inputs)
                prob = float(out[0][0])
                return prob >= self.threshold
            except Exception as e:
                logger.debug("[VAD] ONNX inference error: %s", e)

        # Fallback: Root-Mean-Square (RMS) energy threshold calculation
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        return float(rms) > 0.02

    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray | None:
        """
        Process continuous stream of audio chunks.
        Tracks speech start and speech end events.

        :param audio_chunk: 1D numpy array (float32, 16kHz mono)
        :return: Complete speech segment array when speech ends, else None.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None

        chunk_duration = len(audio_chunk) / self.sample_rate
        speech_detected = self.is_speech(audio_chunk)

        if speech_detected:
            if not self.is_speaking:
                logger.info("[VAD] VAD SPEECH START")
                self.is_speaking = True
                self.speech_chunks.clear()
                self.speech_duration = 0.0
                self.silence_duration = 0.0

            self.speech_chunks.append(audio_chunk)
            self.speech_duration += chunk_duration
            self.silence_duration = 0.0
        else:
            if self.is_speaking:
                self.speech_chunks.append(audio_chunk)
                self.silence_duration += chunk_duration

                if self.silence_duration >= self.min_silence_duration:
                    logger.info("[VAD] VAD SPEECH END")
                    self.is_speaking = False

                    if self.speech_duration >= self.min_speech_duration and self.speech_chunks:
                        speech_segment = np.concatenate(self.speech_chunks)
                        self.speech_chunks.clear()
                        self.speech_duration = 0.0
                        self.silence_duration = 0.0
                        return speech_segment
                    else:
                        self.speech_chunks.clear()
                        self.speech_duration = 0.0
                        self.silence_duration = 0.0

        return None
