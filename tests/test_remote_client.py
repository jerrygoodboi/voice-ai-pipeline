"""
Unit tests for RemotePipelineClient.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import urllib.parse

from src.pipeline.remote_client import RemotePipelineClient


class TestRemotePipelineClient(unittest.TestCase):
    """Test RemotePipelineClient communication."""

    def setUp(self) -> None:
        self.client = RemotePipelineClient(server_url="http://test-server:8000")

    @patch("requests.get")
    def test_check_health_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "status": "healthy",
            "whisper_model": "base",
            "ollama_model": "qwen2.5",
            "ollama_reachable": True,
            "kokoro_voice": "af_heart",
        }
        self.assertTrue(self.client.check_health())
        self.assertTrue(self.client.is_connected)

    @patch("requests.get")
    def test_check_health_offline(self, mock_get: MagicMock) -> None:
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Offline")
        self.assertFalse(self.client.check_health())
        self.assertFalse(self.client.is_connected)

    @patch("requests.post")
    def test_process_speech_success(self, mock_post: MagicMock) -> None:
        fake_audio = np.zeros(24000, dtype=np.float32)
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = fake_audio.tobytes()
        mock_post.return_value.headers = {
            "X-Sample-Rate": "24000",
            "X-Transcription": urllib.parse.quote("Hello assistant"),
            "X-Response-Text": urllib.parse.quote("Hello user"),
        }

        speech_input = np.ones(16000, dtype=np.float32)
        audio_out, sr, trans, resp = self.client.process_speech(speech_input, sample_rate=16000)

        self.assertIsNotNone(audio_out)
        self.assertEqual(len(audio_out), 24000)
        self.assertEqual(sr, 24000)
        self.assertEqual(trans, "Hello assistant")
        self.assertEqual(resp, "Hello user")

    def test_process_speech_empty_input(self) -> None:
        audio_out, sr, trans, resp = self.client.process_speech(np.array([], dtype=np.float32))
        self.assertIsNone(audio_out)
        self.assertEqual(trans, "")
        self.assertEqual(resp, "")


if __name__ == "__main__":
    unittest.main()
