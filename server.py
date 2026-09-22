"""
Voice AI Pipeline Remote Processing Server.
Exposes Whisper STT, Qwen LLM via Ollama, and Kokoro TTS as HTTP endpoints over Tailscale.
"""

import logging
import sys
import urllib.parse
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from config.config import PipelineConfig, get_config
from src.stt.whisper import WhisperSTT, BaseSTT
from src.llm import BaseLLM, create_llm_engine
from src.tts.kokoro import BaseTTS, KokoroTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voice-ai-server")


class PromptRequest(BaseModel):
    prompt: str


class SynthesizeRequest(BaseModel):
    text: str


def create_app(
    config: PipelineConfig | None = None,
    stt: BaseSTT | None = None,
    llm: BaseLLM | None = None,
    tts: BaseTTS | None = None,
) -> FastAPI:
    """Create and configure the FastAPI server application."""
    cfg = config or get_config()

    app = FastAPI(
        title="Voice AI Pipeline Remote Server",
        description="Offload Voice AI processing (Whisper, LLM, Kokoro) over Tailscale",
        version="1.0.0",
    )

    # Initialize server-side AI engines
    logger.info("[Server] Initializing server-side models...")
    _stt = stt or WhisperSTT(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )
    _llm = llm or create_llm_engine(cfg)
    _tts = tts or KokoroTTS(
        voice=cfg.kokoro_voice,
        sample_rate=cfg.sample_rate,
        device=cfg.kokoro_device,
    )

    @app.get("/health")
    async def health_check():
        """Health check returning status of server components."""
        ollama_ok = _llm.check_ollama_status() if hasattr(_llm, "check_ollama_status") else True
        return {
            "status": "healthy",
            "whisper_model": cfg.whisper_model,
            "whisper_device": cfg.whisper_device,
            "llm_provider": cfg.llm_provider,
            "ollama_base_url": cfg.ollama_base_url,
            "ollama_model": cfg.ollama_model,
            "gemini_model": cfg.gemini_model,
            "ollama_reachable": ollama_ok,
            "kokoro_voice": cfg.kokoro_voice,
        }

    @app.post("/process_speech")
    async def process_speech(request: Request, sample_rate: int = 16000):
        """
        End-to-end processing of a speech segment:
        1. Whisper STT
        2. Ollama LLM
        3. Kokoro TTS
        Returns synthesized audio bytes and metadata headers.
        """
        body = await request.body()
        if not body:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        speech_segment = np.frombuffer(body, dtype=np.float32)
        if len(speech_segment) == 0:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # 1. Transcribe speech
        transcription = _stt.transcribe(speech_segment)
        if not transcription or not transcription.strip():
            logger.info("[Server] Silence or empty transcription detected.")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.info("[Server] Transcribed: '%s'", transcription)

        # 2. LLM response generation
        ai_response = _llm.generate_response(transcription)
        if not ai_response or not ai_response.strip():
            logger.warning("[Server] Empty response received from LLM.")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.info("[Server] LLM Response: '%s'", ai_response)

        # 3. TTS audio synthesis
        audio_data, out_sr = _tts.synthesize(ai_response)
        if audio_data is None or len(audio_data) == 0:
            logger.warning("[Server] TTS synthesis produced no audio.")
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        audio_bytes = audio_data.astype(np.float32).tobytes()

        # Safe header encoding for UTF-8 strings
        encoded_transcription = urllib.parse.quote(transcription)
        encoded_ai_response = urllib.parse.quote(ai_response)

        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(out_sr),
            "X-Transcription": encoded_transcription,
            "X-Response-Text": encoded_ai_response,
        }

        return Response(content=audio_bytes, media_type="application/octet-stream", headers=headers)

    @app.post("/transcribe")
    async def transcribe(request: Request, sample_rate: int = 16000):
        """Standalone speech transcription endpoint."""
        body = await request.body()
        if not body:
            return JSONResponse({"text": ""})
        audio_array = np.frombuffer(body, dtype=np.float32)
        text = _stt.transcribe(audio_array)
        return JSONResponse({"text": text})

    @app.post("/generate")
    async def generate(req: PromptRequest):
        """Standalone LLM prompt generation endpoint."""
        response_text = _llm.generate_response(req.prompt)
        return JSONResponse({"response": response_text})

    @app.post("/synthesize")
    async def synthesize(req: SynthesizeRequest):
        """Standalone TTS synthesis endpoint."""
        audio_data, out_sr = _tts.synthesize(req.text)
        if audio_data is None or len(audio_data) == 0:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        audio_bytes = audio_data.astype(np.float32).tobytes()
        return Response(
            content=audio_bytes,
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(out_sr)},
        )

    return app


def main() -> None:
    """Run the Voice AI remote processing server with uvicorn."""
    config = get_config()
    logger.info("Starting Voice AI Pipeline Remote Server...")
    logger.info("  Host: %s", config.server_host)
    logger.info("  Port: %d", config.server_port)
    logger.info("  Whisper Model: %s (%s)", config.whisper_model, config.whisper_device)
    logger.info("  LLM Provider: %s", config.llm_provider)
    if config.llm_provider.lower() == "gemini":
        logger.info("  Gemini Model: %s", config.gemini_model)
    else:
        logger.info("  Ollama Model: %s at %s", config.ollama_model, config.ollama_base_url)
    logger.info("  Kokoro Voice: %s (%s)", config.kokoro_voice, config.kokoro_device)

    app = create_app(config=config)
    uvicorn.run(app, host=config.server_host, port=config.server_port, log_level="info")


if __name__ == "__main__":
    main()
