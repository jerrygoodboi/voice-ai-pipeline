"""
Piper TTS engine using VITS neural models for ultra-fast local CPU synthesis.
Supports direct in-memory Python PiperVoice inference (~0.4s) with CLI subprocess fallback.
"""

import io
import logging
import os
import shutil
import subprocess
import tempfile
import numpy as np

from src.tts.kokoro import BaseTTS

logger = logging.getLogger(__name__)

try:
    from piper import PiperVoice
    PIPER_PYTHON_AVAILABLE = True
except ImportError:
    PIPER_PYTHON_AVAILABLE = False


class PiperTTS(BaseTTS):
    """
    High-speed, offline Text-to-Speech engine using Piper (VITS models).
    Runs in-memory with sub-0.5s latency on modern and low-power CPUs.
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        model_path: str = "",
        sample_rate: int = 22050,
    ) -> None:
        self.voice = voice
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.voice_model = None

        # Resolve model path from config or standard cache
        cache_dir = os.path.expanduser("~/.cache/piper")
        if not self.model_path or not os.path.exists(self.model_path):
            candidate = os.path.join(cache_dir, f"{self.voice}.onnx")
            if os.path.exists(candidate):
                self.model_path = candidate
            elif os.path.exists(os.path.join(cache_dir, "en_US-lessac-medium.onnx")):
                self.model_path = os.path.join(cache_dir, "en_US-lessac-medium.onnx")

        # Load in-memory PiperVoice model for instant <0.5s synthesis
        if PIPER_PYTHON_AVAILABLE and self.model_path and os.path.exists(self.model_path):
            try:
                logger.info("[TTS] Loading Piper model into memory from %s...", self.model_path)
                self.voice_model = PiperVoice.load(self.model_path)
                if hasattr(self.voice_model, "config") and hasattr(self.voice_model.config, "sample_rate"):
                    self.sample_rate = self.voice_model.config.sample_rate
                logger.info("[TTS] Piper in-memory engine ready (sample_rate=%d).", self.sample_rate)
            except Exception as e:
                logger.warning("[TTS] In-memory Piper load failed: %s. Falling back to CLI binary.", e)
                self.voice_model = None

        self.piper_bin = shutil.which("piper")
        if self.piper_bin and os.path.isdir(self.piper_bin):
            piper_exe = os.path.join(self.piper_bin, "piper.exe")
            if os.path.exists(piper_exe):
                self.piper_bin = piper_exe
            else:
                self.piper_bin = None

        if not self.piper_bin:
            # Check local bin or common locations
            home_piper_exe = os.path.expanduser("~/.local/bin/piper/piper.exe")
            if os.path.exists(home_piper_exe):
                self.piper_bin = home_piper_exe
            else:
                home_piper = os.path.expanduser("~/.local/bin/piper")
                if os.path.exists(home_piper) and not os.path.isdir(home_piper):
                    self.piper_bin = home_piper

        if not self.voice_model and not self.piper_bin:
            logger.warning(
                "[TTS] Piper executable not found in PATH. Install via `pip install piper-tts` or download the piper binary."
            )
        else:
            logger.info(
                "[TTS] PiperTTS ready (voice=%s, in_memory=%s).",
                self.voice,
                self.voice_model is not None,
            )

    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """
        Synthesize text into audio samples and sample rate.
        """
        if not text or not text.strip():
            return None, self.sample_rate

        # 1. Ultra-fast in-memory path (~0.4s)
        if self.voice_model is not None:
            try:
                logger.info("[TTS] Synthesizing speech via in-memory Piper (%s)...", self.voice)
                chunks = [c.audio_int16_bytes for c in self.voice_model.synthesize(text.strip())]
                raw_bytes = b"".join(chunks)
                audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                logger.info(
                    "[TTS] Piper generated %d samples (%.2fs at %d Hz).",
                    len(audio_data),
                    len(audio_data) / self.sample_rate,
                    self.sample_rate,
                )
                return audio_data, self.sample_rate
            except Exception as e:
                logger.warning("[TTS] In-memory Piper synthesis error: %s. Trying CLI...", e)

        # 2. Subprocess CLI fallback
        if not self.piper_bin:
            logger.error("[TTS] Piper binary is not installed or available.")
            return None, self.sample_rate

        logger.info("[TTS] Synthesizing speech via Piper CLI (%s)...", self.voice)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name

        try:
            model_arg = self.model_path if self.model_path else self.voice
            cmd = [
                self.piper_bin,
                "--model", model_arg,
                "--output_file", tmp_path,
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            _, err = process.communicate(input=text.strip())

            if process.returncode != 0:
                logger.error("[TTS] Piper synthesis failed: %s", err)
                return None, self.sample_rate

            # Decode wav using ffmpeg
            decode_cmd = [
                "ffmpeg",
                "-loglevel", "error",
                "-i", tmp_path,
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "pipe:1",
            ]
            dec_proc = subprocess.run(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if dec_proc.returncode == 0 and len(dec_proc.stdout) > 0:
                audio_data = np.frombuffer(dec_proc.stdout, dtype=np.float32)
                logger.info("[TTS] PiperTTS generated %d audio samples.", len(audio_data))
                return audio_data, self.sample_rate

            return None, self.sample_rate

        except Exception as e:
            logger.error("[TTS] PiperTTS synthesis error: %s", e)
            return None, self.sample_rate
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
