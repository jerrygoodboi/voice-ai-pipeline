"""
Remote pipeline client for offloading Voice AI processing over Tailscale.
Communicates with the remote Voice AI server to run STT, LLM, and TTS.
"""

import logging
import urllib.parse
import numpy as np
import requests

logger = logging.getLogger(__name__)


class RemotePipelineClient:
    """
    Client for interacting with the remote Voice AI server over Tailscale/HTTP.
    Offloads heavy STT (Whisper), LLM (Qwen/Ollama), and TTS (Kokoro) processing.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ) -> None:
        url = server_url.strip().rstrip("/").rstrip(".")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"

        # If user provided IP or hostname without a port (e.g. http://100.111.239.44), default to 8000
        parsed = urllib.parse.urlparse(url)
        if not parsed.port:
            url = f"{url}:8000"

        self.server_url = url.rstrip("/")
        self.timeout = timeout
        self.is_connected: bool = False

    def check_health(self) -> bool:
        """
        Check if the remote server is reachable and log its configuration.
        """
        url = f"{self.server_url}/health"
        try:
            logger.info("[RemoteClient] Checking connection to remote server at %s...", self.server_url)
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    "[RemoteClient] Connected to remote server at %s. "
                    "Whisper: %s | Ollama: %s (reachable: %s) | Kokoro Voice: %s",
                    self.server_url,
                    data.get("whisper_model", "unknown"),
                    data.get("ollama_model", "unknown"),
                    data.get("ollama_reachable", False),
                    data.get("kokoro_voice", "unknown"),
                )
                self.is_connected = True
                return True
            else:
                logger.warning(
                    "[RemoteClient] Server returned unexpected status code %d: %s",
                    response.status_code,
                    response.text,
                )
                self.is_connected = False
                return False
        except requests.exceptions.ConnectionError:
            logger.warning(
                "[RemoteClient] Could not connect to remote Voice AI server at %s. "
                "Ensure Tailscale is connected and the server is running (`python server.py`).",
                self.server_url,
            )
            self.is_connected = False
            return False
        except Exception as e:
            logger.error("[RemoteClient] Health check failed: %s", e)
            self.is_connected = False
            return False

    def process_speech(
        self,
        speech_segment: np.ndarray,
        sample_rate: int = 16000,
    ) -> tuple[np.ndarray | None, int, str, str]:
        """
        Send speech audio segment to remote server for full pipeline processing:
        Speech -> Whisper STT -> Ollama LLM -> Kokoro TTS -> Synthesized Audio.

        :param speech_segment: 1D numpy array of float32 samples.
        :param sample_rate: Sample rate of the speech segment (default: 16000).
        :return: Tuple of (synthesized_audio_array, sample_rate, transcription, response_text).
        """
        if speech_segment is None or len(speech_segment) == 0:
            return None, sample_rate, "", ""

        url = f"{self.server_url}/process_speech?sample_rate={sample_rate}"
        payload = speech_segment.astype(np.float32).tobytes()

        logger.info(
            "[RemoteClient] Sending %d speech samples (%.2fs) to remote server...",
            len(speech_segment),
            len(speech_segment) / sample_rate,
        )

        try:
            response = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )

            if response.status_code == 200:
                raw_transcription = response.headers.get("X-Transcription", "")
                raw_ai_response = response.headers.get("X-Response-Text", "")
                transcription = urllib.parse.unquote(raw_transcription)
                ai_response = urllib.parse.unquote(raw_ai_response)
                out_sr = int(response.headers.get("X-Sample-Rate", str(sample_rate)))

                audio_bytes = response.content
                if audio_bytes:
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    logger.info(
                        "[RemoteClient] Received %d audio samples (%d Hz) from remote server.",
                        len(audio_data),
                        out_sr,
                    )
                    return audio_data, out_sr, transcription, ai_response
                else:
                    logger.warning("[RemoteClient] Server returned empty audio body.")
                    return None, out_sr, transcription, ai_response
            elif response.status_code == 204:
                # No speech or empty transcription
                logger.info("[RemoteClient] Remote server detected no speech in segment.")
                return None, sample_rate, "", ""
            else:
                logger.error(
                    "[RemoteClient] Remote server error (%d): %s",
                    response.status_code,
                    response.text,
                )
                return None, sample_rate, "", ""
        except requests.exceptions.Timeout:
            logger.error("[RemoteClient] Request to %s timed out after %.1fs.", self.server_url, self.timeout)
            return None, sample_rate, "", ""
        except requests.exceptions.ConnectionError as err:
            logger.error(
                "[RemoteClient] Lost connection to remote server at %s. (Detail: %s). Check if server is running and port is open.",
                self.server_url,
                err,
            )
            return None, sample_rate, "", ""
        except Exception as e:
            logger.error("[RemoteClient] Failed to process speech remotely: %s", e)
            return None, sample_rate, "", ""

    def transcribe(self, speech_segment: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe speech audio segment using remote Whisper STT."""
        url = f"{self.server_url}/transcribe?sample_rate={sample_rate}"
        payload = speech_segment.astype(np.float32).tobytes()
        try:
            resp = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("text", "")
            return ""
        except Exception as e:
            logger.error("[RemoteClient] Transcribe error: %s", e)
            return ""

    def generate(self, prompt: str) -> str:
        """Generate LLM response using remote Ollama."""
        url = f"{self.server_url}/generate"
        try:
            resp = requests.post(url, json={"prompt": prompt}, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("response", "")
            return ""
        except Exception as e:
            logger.error("[RemoteClient] Generate error: %s", e)
            return ""

    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        """Synthesize text into speech audio using remote Kokoro TTS."""
        url = f"{self.server_url}/synthesize"
        try:
            resp = requests.post(url, json={"text": text}, timeout=self.timeout)
            if resp.status_code == 200:
                sr = int(resp.headers.get("X-Sample-Rate", "24000"))
                audio_data = np.frombuffer(resp.content, dtype=np.float32)
                return audio_data, sr
            return None, 24000
        except Exception as e:
            logger.error("[RemoteClient] Synthesize error: %s", e)
            return None, 24000
