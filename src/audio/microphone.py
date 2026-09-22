"""
Microphone audio capture module.
Primary backend: parec (PipeWire/PulseAudio) subprocess — works on LFS + PipeWire.
Fallback backend: sounddevice/PortAudio.
Provides a clean abstraction for capturing real-time mono audio streams.
"""

from abc import ABC, abstractmethod
import logging
import os
import queue
import shutil
import subprocess
import threading
from typing import Any
import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # Fallback unavailable

logger = logging.getLogger(__name__)


class BaseAudioInput(ABC):
    """Abstract interface for audio input devices."""

    @abstractmethod
    def start(self) -> None:
        """Start capturing audio."""
        pass

    @abstractmethod
    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Read a chunk of audio data.

        :param timeout: Maximum time in seconds to wait for data.
        :return: 1D numpy array of audio samples (float32) or None if timeout.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release all audio stream resources."""
        pass


class PipewireMicrophone(BaseAudioInput):
    """
    Microphone capture using parec (PipeWire PulseAudio compat layer).
    Works on LFS + PipeWire without needing PortAudio to enumerate capture devices.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
        device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device = device  # optional parec --device=<name>

        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_running: bool = False

    def _reader_thread(self) -> None:
        """Background thread: reads raw float32 bytes from parec stdout into queue."""
        bytes_per_frame = self.channels * 4  # float32 = 4 bytes
        chunk_bytes = self.chunk_size * bytes_per_frame
        try:
            while self._is_running and self._proc and self._proc.poll() is None:
                raw = self._proc.stdout.read(chunk_bytes)
                if not raw or len(raw) < chunk_bytes:
                    break
                chunk = np.frombuffer(raw, dtype=np.float32)
                if self.channels > 1:
                    chunk = chunk.reshape(-1, self.channels).mean(axis=1)  # downmix to mono
                self._audio_queue.put(chunk)
        except Exception as e:
            if self._is_running:
                logger.error("[AUDIO] parec reader thread error: %s", e)

    def start(self) -> None:
        """Start parec subprocess and background reader thread."""
        if self._is_running:
            return

        cmd = [
            "parec",
            "--raw",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
            "--format=float32le",
        ]
        if self.device:
            cmd.append(f"--device={self.device}")

        # Inherit env and inject PipeWire/PulseAudio socket paths if on POSIX
        env = os.environ.copy()
        if hasattr(os, "getuid"):
            uid = os.getuid()
            env.setdefault("PULSE_RUNTIME_PATH", f"/run/user/{uid}/pulse")
            env.setdefault("PIPEWIRE_RUNTIME_DIR", f"/run/user/{uid}")
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")

        logger.info("[AUDIO] Starting PipeWire mic via parec: %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            self._is_running = True
            self._thread = threading.Thread(target=self._reader_thread, daemon=True)
            self._thread.start()
            logger.info("[AUDIO] PipeWire mic stream started (rate=%d, chunk=%d).", self.sample_rate, self.chunk_size)
        except FileNotFoundError:
            raise RuntimeError(
                "parec not found. Ensure pipewire-pulse or pulseaudio is installed."
            )
        except Exception as e:
            self._is_running = False
            logger.error("[AUDIO] Failed to start parec: %s", e)
            raise

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read next audio chunk from queue."""
        if not self._is_running:
            return None
        try:
            return self._audio_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop the parec process and reader thread."""
        if not self._is_running:
            return
        logger.info("[AUDIO] Stopping PipeWire mic stream...")
        self._is_running = False
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("[AUDIO] PipeWire mic stream stopped.")

    def close(self) -> None:
        """Release resources."""
        self.stop()

    def __enter__(self) -> "PipewireMicrophone":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class Microphone(BaseAudioInput):
    """
    Microphone capture implementation using sounddevice (PortAudio).
    Captures mono audio into a thread-safe queue.
    Falls back to PipewireMicrophone if no PortAudio input device is found.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        channels: int = 1,
        dtype: str = "float32",
        device: int | str | None = None,
    ) -> None:
        """
        Initialize Microphone instance.

        :param sample_rate: Audio sampling rate in Hz (default: 16000).
        :param chunk_size: Number of frames per block/chunk (default: 512).
        :param channels: Number of audio channels (1 for mono).
        :param dtype: Audio sample data type ('float32' or 'int16').
        :param device: Optional input device index or name.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = dtype
        self.device = device

        self._stream: Any = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_running: bool = False

        # Detect if PortAudio has any viable input device; fall back to parec
        self._delegate: BaseAudioInput | None = self._detect_backend()

    def _detect_backend(self) -> "BaseAudioInput | None":
        """Check if sounddevice has a working input device. Return PipewireMicrophone if not and parec is present."""
        has_parec = shutil.which("parec") is not None
        if sd is None:
            if has_parec:
                logger.info("[AUDIO] sounddevice unavailable. Using PipeWire (parec) backend.")
                return PipewireMicrophone(
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    channels=self.channels,
                )
            logger.warning("[AUDIO] sounddevice is not available and parec was not found.")
            return None
        try:
            devs = sd.query_devices()
            has_input = any(d.get("max_input_channels", 0) > 0 for d in devs)
            if not has_input and has_parec:
                logger.info(
                    "[AUDIO] No PortAudio input devices found. Using PipeWire (parec) backend."
                )
                return PipewireMicrophone(
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    channels=self.channels,
                )
        except Exception as e:
            if has_parec:
                logger.warning("[AUDIO] PortAudio device query failed (%s). Using PipeWire backend.", e)
                return PipewireMicrophone(
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    channels=self.channels,
                )
            logger.warning("[AUDIO] PortAudio device query failed: %s", e)
        return None  # PortAudio has input devices — use native sounddevice path

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback invoked by sounddevice for each audio block."""
        if status:
            logger.warning("[AUDIO] Stream status warning: %s", status)
        if self._is_running:
            chunk = indata.copy().flatten()
            self._audio_queue.put(chunk)

    def start(self) -> None:
        """Start recording from microphone."""
        if self._delegate:
            return self._delegate.start()

        if sd is None:
            raise RuntimeError(
                "sounddevice package is not installed. Please run `pip install sounddevice numpy`."
            )
        if self._is_running:
            return

        try:
            logger.info(
                "[AUDIO] Starting microphone stream (sample_rate=%d, chunk_size=%d, channels=%d)...",
                self.sample_rate, self.chunk_size, self.channels,
            )
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            logger.info("[AUDIO] Microphone stream started successfully.")
        except Exception as e:
            logger.error("[AUDIO] Failed to start microphone stream: %s", e)
            self._is_running = False
            raise

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Read the next audio chunk from queue."""
        if self._delegate:
            return self._delegate.read_chunk(timeout=timeout)

        if not self._is_running:
            logger.warning("[AUDIO] Cannot read chunk: Microphone stream is not running.")
            return None
        try:
            return self._audio_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            logger.debug("[AUDIO] Read chunk timed out after %.2f seconds.", timeout)
            return None

    def stop(self) -> None:
        """Stop recording audio stream."""
        if self._delegate:
            return self._delegate.stop()

        if not self._is_running:
            return
        logger.info("[AUDIO] Stopping microphone stream...")
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error("[AUDIO] Error while closing microphone stream: %s", e)
            finally:
                self._stream = None
        logger.info("[AUDIO] Microphone stream stopped.")

    def close(self) -> None:
        """Release microphone resources."""
        self.stop()

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
