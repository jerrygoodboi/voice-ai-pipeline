"""
Piper TTS engine using VITS neural models for ultra-fast local CPU synthesis.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import numpy as np

from src.tts.kokoro import BaseTTS

logger = logging.getLogger(__name__)


class PiperTTS(BaseTTS):
    """
    High-speed, offline Text-to-Speech engine using Piper (VITS models).
    Runs with near-zero latency on modern and low-power CPUs.
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

        self.piper_bin = shutil.which("piper")
        if not self.piper_bin:
            # Check local bin or common locations
            home_piper = os.path.expanduser("~/.local/bin/piper")
            if os.path.exists(home_piper):
                self.piper_bin = home_piper

        if not self.piper_bin:
            logger.warning(
                "[TTS] Piper executable not found in PATH. Install via `pip install piper-tts` or download the piper binary."
            )
        else:
            logger.info("[TTS] PiperTTS initialized using binary at %s (voice=%s).", self.piper_bin, self.voice)

    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """
        Synthesize text into audio samples and sample rate.
        """
        if not text or not text.strip():
            return None, self.sample_rate

        if not self.piper_bin:
            logger.error("[TTS] Piper binary is not installed or available.")
            return None, self.sample_rate

        logger.info("[TTS] Synthesizing speech via PiperTTS (%s)...", self.voice)

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
