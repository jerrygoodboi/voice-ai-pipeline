"""
Unit tests for Voice AI Pipeline remote server.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import urllib.parse
from fastapi.testclient import TestClient

from config.config import PipelineConfig
from server import create_app
from src.stt.whisper import BaseSTT
from src.llm.qwen import BaseLLM
from src.tts.kokoro import BaseTTS


class MockSTT(BaseSTT):
    def transcribe(self, audio_data: np.ndarray | bytes) -> str:
        return "mock transcription"


class MockLLM(BaseLLM):
    def generate_response(self, prompt: str) -> str:
        return f"mock answer to {prompt}"

    def check_ollama_status(self) -> bool:
        return True


class MockTTS(BaseTTS):
    def synthesize(self, text: str) -> tuple[np.ndarray | None, int]:
        # Return 1 second of audio at 24000 Hz
        return np.ones(24000, dtype=np.float32), 24000


class TestServerEndpoints(unittest.TestCase):
    """Test server API endpoints with mocked engines."""

    def setUp(self) -> None:
        self.config = PipelineConfig(
            whisper_model="mock-whisper",
            ollama_model="mock-qwen",
            kokoro_voice="mock-voice",
        )
        self.app = create_app(
            config=self.config,
            stt=MockSTT(),
            llm=MockLLM(),
            tts=MockTTS(),
        )
        self.client = TestClient(self.app)

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["whisper_model"], "mock-whisper")
        self.assertEqual(data["ollama_model"], "mock-qwen")

    def test_process_speech(self) -> None:
        fake_speech = np.ones(16000, dtype=np.float32)
        response = self.client.post(
            "/process_speech?sample_rate=16000",
            content=fake_speech.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate"], "24000")
        self.assertEqual(urllib.parse.unquote(response.headers["X-Transcription"]), "mock transcription")
        self.assertEqual(urllib.parse.unquote(response.headers["X-Response-Text"]), "mock answer to mock transcription")

        audio_received = np.frombuffer(response.content, dtype=np.float32)
        self.assertEqual(len(audio_received), 24000)

    def test_process_speech_empty(self) -> None:
        response = self.client.post(
            "/process_speech",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 204)

    def test_transcribe_endpoint(self) -> None:
        fake_audio = np.ones(16000, dtype=np.float32)
        response = self.client.post("/transcribe", content=fake_audio.tobytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "mock transcription"})

    def test_generate_endpoint(self) -> None:
        response = self.client.post("/generate", json={"prompt": "what is AI?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "mock answer to what is AI?"})

    def test_generate_endpoint_with_interrupted_context(self) -> None:
        payload = {
            "prompt": "explain machine learning",
            "interrupted_context": {
                "previousUserPrompt": "what is AI?",
                "previousAssistantText": "AI is artificial intelligence...",
                "wasInterrupted": True
            }
        }
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertTrue("response" in response.json())

    def test_generate_endpoint_with_empty_interrupted_context(self) -> None:
        payload = {
            "prompt": "what is deep learning?",
            "interrupted_context": {}
        }
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "mock answer to what is deep learning?"})

    def test_synthesize_endpoint(self) -> None:
        response = self.client.post("/synthesize", json={"text": "hello world"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate"], "24000")
        audio = np.frombuffer(response.content, dtype=np.float32)
        self.assertEqual(len(audio), 24000)


if __name__ == "__main__":
    unittest.main()
