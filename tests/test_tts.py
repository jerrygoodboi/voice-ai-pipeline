"""
Unit tests for Kokoro TTS module.
"""

import unittest
from src.tts.kokoro import KokoroTTS


class TestKokoroTTS(unittest.TestCase):
    """Test Kokoro TTS audio synthesis."""

    def test_tts_init(self) -> None:
        tts = KokoroTTS(voice="af_heart", sample_rate=16000)
        self.assertEqual(tts.voice, "af_heart")

    def test_tts_synthesize_text(self) -> None:
        tts = KokoroTTS()
        audio, sr = tts.synthesize("Test audio synthesis")
        if audio is None:
            self.skipTest("Neither kokoro-onnx nor pyttsx3 is installed in this test environment")
        self.assertIsNotNone(audio)
        self.assertGreater(len(audio), 0)
        self.assertGreater(sr, 0)

    def test_tts_empty_text(self) -> None:
        tts = KokoroTTS()
        audio, sr = tts.synthesize("")
        self.assertIsNone(audio)


if __name__ == "__main__":
    unittest.main()
