"""
Edge TTS engine using Microsoft Azure Neural voices via edge-tts.
Provides studio-grade speech synthesis with zero local CPU overhead.
Optimized for IPv4 to eliminate Linux IPv6 routing delays.
"""

import asyncio
import io
import logging
import socket
import subprocess
import numpy as np

from src.tts.kokoro import BaseTTS

logger = logging.getLogger(__name__)

try:
    import aiohttp
    # Prevent Linux IPv6 DNS/routing hang in aiohttp
    _orig_tcp_init = aiohttp.TCPConnector.__init__

    def _ipv4_tcp_init(self, *args, **kwargs):
        kwargs["family"] = socket.AF_INET
        _orig_tcp_init(self, *args, **kwargs)

    aiohttp.TCPConnector.__init__ = _ipv4_tcp_init
except Exception:
    pass

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class EdgeTTS(BaseTTS):
    """
    High-quality Text-to-Speech using edge-tts.
    Connects to Microsoft Azure Neural endpoints and decodes audio directly to floating point arrays.
    """

    def __init__(
        self,
        voice: str = "en-US-JennyNeural",
        sample_rate: int = 16000,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.rate = rate
        self.pitch = pitch
        logger.info("[TTS] EdgeTTS initialized with voice '%s' (sample_rate=%d).", self.voice, self.sample_rate)

    async def _synthesize_async(self, text: str) -> bytes:
        """Asynchronously stream audio chunks from edge-tts."""
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        return audio_stream.getvalue()

    def _decode_mp3_to_numpy(self, mp3_bytes: bytes) -> np.ndarray | None:
        """Decode MP3 bytes to 1D float32 numpy array using ffmpeg."""
        if not mp3_bytes:
            return None

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "pipe:1",
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = process.communicate(input=mp3_bytes)
            if process.returncode != 0:
                logger.error("[TTS] ffmpeg decoding error: %s", err.decode(errors="ignore"))
                return None

            audio_data = np.frombuffer(out, dtype=np.float32)
            return audio_data
        except Exception as e:
            logger.error("[TTS] Failed to decode audio stream: %s", e)
            return None

    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """
        Synthesize text into audio samples and sample rate.

        :param text: Text string to synthesize
        :return: Tuple of (1D float32 numpy array, sample_rate)
        """
        if not text or not text.strip():
            return None, self.sample_rate

        if not EDGE_TTS_AVAILABLE:
            logger.error("[TTS] edge-tts package is not installed. Run `pip install edge-tts`.")
            return None, self.sample_rate

        logger.info("[TTS] Synthesizing speech via EdgeTTS (%s)...", self.voice)
        try:
            mp3_bytes = asyncio.run(self._synthesize_async(text.strip()))
            audio_data = self._decode_mp3_to_numpy(mp3_bytes)

            if audio_data is not None and len(audio_data) > 0:
                logger.info(
                    "[TTS] EdgeTTS generated %d samples (%.2fs at %d Hz).",
                    len(audio_data),
                    len(audio_data) / self.sample_rate,
                    self.sample_rate,
                )
                return audio_data, self.sample_rate
            else:
                logger.warning("[TTS] EdgeTTS decoding produced empty audio.")
                return None, self.sample_rate

        except Exception as e:
            logger.error("[TTS] EdgeTTS synthesis error: %s", e)
            return None, self.sample_rate
