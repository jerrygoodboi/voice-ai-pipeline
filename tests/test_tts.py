"""
Unit tests for TTS modules (EdgeTTS, PiperTTS, KokoroTTS, and factory).
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.config import PipelineConfig
from src.tts.kokoro import KokoroTTS
from src.tts.edge import EdgeTTS
from src.tts.piper import PiperTTS
from src.tts.factory import get_tts_engine


class TestTTSModules(unittest.TestCase):
    """Test suite for Text-to-Speech synthesis engines and factory."""

    def test_kokoro_init(self) -> None:
        tts = KokoroTTS(voice="af_heart", sample_rate=16000)
        self.assertEqual(tts.voice, "af_heart")

    def test_kokoro_empty_text(self) -> None:
        tts = KokoroTTS()
        audio, sr = tts.synthesize("")
        self.assertIsNone(audio)

    def test_edge_tts_init(self) -> None:
        tts = EdgeTTS(voice="en-US-JennyNeural", sample_rate=16000)
        self.assertEqual(tts.voice, "en-US-JennyNeural")
        self.assertEqual(tts.sample_rate, 16000)

    def test_edge_tts_empty_text(self) -> None:
        tts = EdgeTTS()
        audio, sr = tts.synthesize("")
        self.assertIsNone(audio)

    @patch("src.tts.edge.EdgeTTS._synthesize_async")
    @patch("src.tts.edge.EdgeTTS._decode_mp3_to_numpy")
    def test_edge_tts_synthesize_mocked(self, mock_decode: MagicMock, mock_synth: MagicMock) -> None:
        mock_synth.return_value = b"fake_mp3_data"
        mock_decode.return_value = np.zeros(16000, dtype=np.float32)

        tts = EdgeTTS(voice="en-US-JennyNeural")
        audio, sr = tts.synthesize("Hello world")
        self.assertIsNotNone(audio)
        self.assertEqual(len(audio), 16000)
        self.assertEqual(sr, 16000)

    def test_piper_tts_init(self) -> None:
        tts = PiperTTS(voice="en_US-lessac-medium", sample_rate=22050)
        self.assertEqual(tts.voice, "en_US-lessac-medium")

    def test_piper_empty_text(self) -> None:
        tts = PiperTTS()
        audio, sr = tts.synthesize("")
        self.assertIsNone(audio)

    def test_factory_switching(self) -> None:
        cfg_edge = PipelineConfig(tts_engine="edge")
        engine_edge = get_tts_engine(cfg_edge)
        self.assertIsInstance(engine_edge, EdgeTTS)

        cfg_piper = PipelineConfig(tts_engine="piper")
        engine_piper = get_tts_engine(cfg_piper)
        self.assertIsInstance(engine_piper, PiperTTS)

        cfg_kokoro = PipelineConfig(tts_engine="kokoro")
        engine_kokoro = get_tts_engine(cfg_kokoro)
        self.assertIsInstance(engine_kokoro, KokoroTTS)


if __name__ == "__main__":
    unittest.main()
