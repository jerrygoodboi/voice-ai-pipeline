"""
Audio output module for playing audio through system speakers.
Primary backend: paplay (PipeWire/PulseAudio) — works on LFS + PipeWire.
Fallback backend: sounddevice/PortAudio.
"""

from abc import ABC, abstractmethod
import io
import logging
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Any
import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

try:
    import soundfile as sf
except (ImportError, OSError):
    sf = None

logger = logging.getLogger(__name__)


class BaseAudioOutput(ABC):
    """Abstract interface for audio output devices."""

    @abstractmethod
    def play(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """
        Play audio data through speakers.

        :param audio_data: Audio samples (1D numpy float32 or int16 array)
        :param sample_rate: Audio sampling rate in Hz
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop playback immediately."""
        pass


class PipeWireSpeaker(BaseAudioOutput):
    """
    Speaker output using paplay (PipeWire PulseAudio compat layer).
    Writes audio as a WAV to stdout and pipes to paplay.
    Works on LFS + PipeWire without needing PortAudio output device.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = device  # optional paplay --device=<sink-name>
        self._proc: subprocess.Popen | None = None

    def _make_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert float32 numpy array to WAV bytes (in-memory)."""
        audio = audio.astype(np.float32)
        num_samples = len(audio)
        num_channels = 1
        bits_per_sample = 32
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,          # subchunk1 size
            3,           # PCM float = 3
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + audio.tobytes()

    def play(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """Play audio via paplay (PipeWire)."""
        if audio_data is None or len(audio_data) == 0:
            logger.warning("[AUDIO] Received empty audio buffer for playback.")
            return

        env = os.environ.copy()
        if hasattr(os, "getuid"):
            uid = os.getuid()
            env.setdefault("PULSE_RUNTIME_PATH", f"/run/user/{uid}/pulse")
            env.setdefault("PIPEWIRE_RUNTIME_DIR", f"/run/user/{uid}")
            env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")

        cmd = [
            "paplay",
            "--raw",
            f"--rate={sample_rate}",
            "--channels=1",
            "--format=float32le",
        ]
        if self.device:
            cmd.append(f"--device={self.device}")

        logger.info("[AUDIO] Playing response via PipeWire paplay (%d samples, %d Hz)...", len(audio_data), sample_rate)
        try:
            raw = audio_data.astype(np.float32).tobytes()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            proc.communicate(input=raw)
            proc.wait()
            logger.info("[AUDIO] PipeWire audio playback finished.")
        except FileNotFoundError:
            logger.error("[AUDIO] paplay not found. Install pipewire-pulse.")
        except Exception as e:
            logger.error("[AUDIO] PipeWire speaker playback error: %s", e)

    def stop(self) -> None:
        """Stop current playback."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None
            logger.info("[AUDIO] PipeWire playback stopped.")


class Speaker(BaseAudioOutput):
    """
    Speaker audio output implementation.
    Auto-detects backend: uses PipeWireSpeaker always on PipeWire systems,
    falls back to sounddevice if ALSA output is available.
    """

    def __init__(self, device: int | str | None = None) -> None:
        """
        Initialize Speaker output device.

        :param device: Optional output device index or name.
        """
        self.device = device
        self._is_playing: bool = False
        self._delegate: BaseAudioOutput = self._detect_backend()

    def _detect_backend(self) -> BaseAudioOutput:
        """Prefer PipeWire (paplay) over sounddevice/PortAudio."""
        # Check if paplay is available (PipeWire/PulseAudio compat)
        if shutil.which("paplay") is not None:
            logger.info("[AUDIO] Using PipeWire (paplay) output backend.")
            return PipeWireSpeaker(device=self.device if isinstance(self.device, str) else None)

        logger.info("[AUDIO] paplay not found. Falling back to sounddevice output.")
        return self  # will use sounddevice path below (no delegate)

    def play(self, audio_data: np.ndarray, sample_rate: int = 16000) -> None:
        """Play audio through detected backend."""
        if self._delegate is not self:
            return self._delegate.play(audio_data, sample_rate)

        # sounddevice fallback
        if sd is None:
            logger.error("[AUDIO] sounddevice is not available. Playback skipped.")
            return
        if audio_data is None or len(audio_data) == 0:
            logger.warning("[AUDIO] Received empty audio buffer for playback.")
            return
        try:
            logger.info("[AUDIO] Playing response (%d samples, %d Hz)...", len(audio_data), sample_rate)
            self._is_playing = True
            sd.play(audio_data, samplerate=sample_rate, device=self.device)
            sd.wait()
            logger.info("[AUDIO] Audio playback finished.")
        except Exception as e:
            logger.error("[AUDIO] Speaker playback error: %s", e)
        finally:
            self._is_playing = False

    def stop(self) -> None:
        """Stop current playback."""
        if self._delegate is not self:
            return self._delegate.stop()
        if sd is not None and self._is_playing:
            try:
                sd.stop()
                logger.info("[AUDIO] Speaker playback stopped.")
            except Exception as e:
                logger.error("[AUDIO] Error stopping speaker playback: %s", e)
            finally:
                self._is_playing = False
