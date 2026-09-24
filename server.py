"""
Voice AI Pipeline Remote Processing Server.
Exposes Whisper STT, Qwen LLM via Ollama, and Kokoro TTS as HTTP endpoints over Tailscale.
"""

import asyncio
import io
import json
import logging
import sys
import subprocess
import urllib.parse
import wave
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn

from config.config import PipelineConfig, get_config
from src.stt.whisper import WhisperSTT, BaseSTT
from src.llm import BaseLLM, create_llm_engine
from src.tts.kokoro import BaseTTS, KokoroTTS

from src.tts.factory import get_tts_engine
from src.llm.session import SessionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("voice-ai-server")


class PromptRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    interrupted_context: dict | None = None


class SessionResetRequest(BaseModel):
    session_id: str


class SynthesizeRequest(BaseModel):
    text: str


def decode_input_audio(body: bytes, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode incoming audio bytes (raw float32, WAV, WebM, OGG, MP3) to 1D float32 numpy array.
    """
    if not body:
        return np.array([], dtype=np.float32)

    # 1. Try Python wave module for standard WAV headers
    if body.startswith(b"RIFF") and b"WAVE" in body[:12]:
        try:
            with wave.open(io.BytesIO(body), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                if sampwidth == 2:
                    raw_int16 = np.frombuffer(frames, dtype=np.int16)
                    if n_channels > 1:
                        raw_int16 = raw_int16[::n_channels]
                    return raw_int16.astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    raw_f32 = np.frombuffer(frames, dtype=np.float32)
                    if n_channels > 1:
                        raw_f32 = raw_f32[::n_channels]
                    return raw_f32
        except Exception as e:
            logger.debug("[Server] Wave decode exception: %s", e)

    # 2. Check for container magic headers (WebM, OGG, MP3, RIFF) -> decode via ffmpeg FIRST
    is_container = (
        body.startswith(b"\x1a\x45\xdf\xa3")  # WebM / EBML
        or body.startswith(b"OggS")            # OGG
        or body.startswith(b"ID3")             # MP3 ID3
        or body.startswith(b"\xff\xfb")        # MP3 sync frame
        or body.startswith(b"RIFF")            # RIFF container
    )

    if is_container:
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(sample_rate),
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate(input=body)
            if proc.returncode == 0 and len(out) > 0:
                return np.frombuffer(out, dtype=np.float32)
        except Exception as e:
            logger.debug("[Server] ffmpeg decode exception: %s", e)

    # 3. Try direct float32 array interpretation for raw headerless PCM stream
    try:
        arr = np.frombuffer(body, dtype=np.float32)
        if len(arr) > 0 and not np.isnan(arr).any() and np.max(np.abs(arr)) <= 1.0:
            return arr
    except Exception:
        pass

    # 4. Fallback ffmpeg attempt for any unrecognized audio format
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(input=body)
        if proc.returncode == 0 and len(out) > 0:
            return np.frombuffer(out, dtype=np.float32)
    except Exception as e:
        logger.debug("[Server] Fallback ffmpeg decode exception: %s", e)

    return np.array([], dtype=np.float32)



from src.vad.silero_vad import BaseVAD, SileroVAD


def create_app(
    config: PipelineConfig | None = None,
    vad: BaseVAD | None = None,
    stt: BaseSTT | None = None,
    llm: BaseLLM | None = None,
    tts: BaseTTS | None = None,
) -> FastAPI:
    """Create and configure the FastAPI server application."""
    cfg = config or get_config()

    app = FastAPI(
        title="Voice AI Pipeline Remote Server",
        description="Offload Voice AI processing (Whisper, LLM, Piper/Kokoro) over Tailscale & Web UI",
        version="1.0.0",
    )

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Sample-Rate", "X-Transcription", "X-Response-Text"],
    )

    # Initialize server-side AI engines
    logger.info("[Server] Initializing server-side models...")
    _vad = vad or SileroVAD(
        threshold=cfg.vad_threshold,
        min_speech_duration=cfg.vad_min_speech_duration,
        min_silence_duration=cfg.vad_min_silence_duration,
        sample_rate=cfg.sample_rate,
        device=cfg.vad_device,
    )
    _stt = stt or WhisperSTT(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )
    _llm = llm or create_llm_engine(cfg)
    _tts = tts or get_tts_engine(cfg)
    _session_manager = SessionManager(max_tokens=1500)

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
        1. Silero VAD
        2. Whisper STT
        3. Gemini / Qwen LLM
        4. Piper / Kokoro TTS
        Returns synthesized audio bytes and metadata headers.
        """
        body = await request.body()
        if not body:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.info("[AUDIO] AUDIO RECEIVED (%d bytes)", len(body))

        audio_chunk = decode_input_audio(body, sample_rate)
        if len(audio_chunk) == 0:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # Process audio chunk through Silero VAD
        chunk_size = 512
        speech_segment = None
        for i in range(0, len(audio_chunk), chunk_size):
            sub_chunk = audio_chunk[i : i + chunk_size]
            if len(sub_chunk) == chunk_size:
                seg = _vad.process_chunk(sub_chunk)
                if seg is not None:
                    speech_segment = seg

        if speech_segment is None:
            if _vad.speech_chunks:
                speech_segment = np.concatenate(_vad.speech_chunks)
                _vad.reset()
            elif len(audio_chunk) >= int(sample_rate * cfg.vad_min_speech_duration):
                speech_segment = audio_chunk

        if speech_segment is None or len(speech_segment) == 0:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # 1. Transcribe speech with Whisper STT
        transcription = _stt.transcribe(speech_segment)
        if not transcription or not transcription.strip():
            logger.info("[Server] Silence or empty transcription detected.")
            _vad.reset()
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.info("[STT] TRANSCRIPTION: '%s'", transcription)

        # 2. LLM response generation with session history
        session_id = request.headers.get("X-Session-ID", "default")
        history = _session_manager.get_history(session_id)
        ai_response = _llm.generate_response(transcription, history=history)
        if not ai_response or not ai_response.strip():
            logger.warning("[Server] Empty response received from LLM.")
            _vad.reset()
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        _session_manager.add_turn(session_id, "user", transcription)
        if ai_response and not ai_response.startswith("[Gemini Error"):
            _session_manager.add_turn(session_id, "model", ai_response)

        logger.info("[LLM] GEMINI RESPONSE: '%s'", ai_response)

        # 3. TTS audio synthesis
        audio_data, out_sr = _tts.synthesize(ai_response)
        if audio_data is None or len(audio_data) == 0:
            logger.warning("[Server] TTS synthesis produced no audio.")
            _vad.reset()
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        logger.info("[TTS] PIPER SYNTHESIS: %d samples", len(audio_data))
        _vad.reset()

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

    @app.websocket("/ws/transcribe")
    async def websocket_transcribe(websocket: WebSocket):
        """
        Real-time streaming speech transcription endpoint over WebSocket.
        Receives binary PCM16 audio chunks (16000Hz mono).
        Emits live word-by-word interim updates and finalized transcripts.
        """
        await websocket.accept()
        logger.info("[WebSocket] Real-time STT client connected.")
        pcm_chunks = []
        total_samples = 0
        is_transcribing = False
        import time
        last_transcribe_time = 0.0
        loop = asyncio.get_event_loop()

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    data_bytes = message["bytes"]
                    if len(data_bytes) >= 2:
                        chunk_f32 = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        pcm_chunks.append(chunk_f32)
                        total_samples += len(chunk_f32)

                    # Limit maximum rolling window to last 6 seconds (96000 samples)
                    if total_samples > 96000:
                        combined = np.concatenate(pcm_chunks)[-96000:]
                        pcm_chunks = [combined]
                        total_samples = len(combined)

                    # Trigger sliding window transcribe when audio >= 0.5s (8000 samples) and throttled to every 500ms
                    now = time.time()
                    if not is_transcribing and total_samples >= 8000 and (now - last_transcribe_time >= 0.5):
                        is_transcribing = True
                        last_transcribe_time = now
                        try:
                            combined = np.concatenate(pcm_chunks)
                            text = await loop.run_in_executor(None, _stt.transcribe, combined)
                            if text and text.strip():
                                await websocket.send_json({"type": "interim", "text": text.strip()})
                        except Exception as e:
                            logger.debug("[WebSocket STT] Interim error: %s", e)
                        finally:
                            is_transcribing = False

                elif "text" in message and message["text"]:
                    try:
                        msg_json = json.loads(message["text"])
                        msg_type = msg_json.get("type")
                        if msg_type == "finalize":
                            final_text = ""
                            if pcm_chunks:
                                combined = np.concatenate(pcm_chunks)
                                if len(combined) >= 4000:
                                    final_text = await loop.run_in_executor(None, _stt.transcribe, combined)
                            await websocket.send_json({"type": "final", "text": (final_text or "").strip()})
                            pcm_chunks.clear()
                            total_samples = 0
                            is_transcribing = False
                        elif msg_type == "reset":
                            pcm_chunks.clear()
                            total_samples = 0
                            is_transcribing = False
                    except Exception as e:
                        logger.debug("[WebSocket STT] Message error: %s", e)

        except WebSocketDisconnect:
            logger.info("[WebSocket] Real-time STT client disconnected.")
        except Exception as e:
            if "disconnect" not in str(e).lower():
                logger.warning("[WebSocket] Connection error: %s", e)
        finally:
            pcm_chunks.clear()
            total_samples = 0
            is_transcribing = False

    @app.post("/transcribe")
    async def transcribe(request: Request, sample_rate: int = 16000):
        """Standalone speech transcription endpoint."""
        body = await request.body()
        if not body:
            return JSONResponse({"text": ""})
        audio_array = decode_input_audio(body, sample_rate)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _stt.transcribe, audio_array)
        return JSONResponse({"text": text})

    @app.post("/generate")
    async def generate(req: PromptRequest):
        """Standalone LLM prompt generation endpoint with in-memory multi-turn session history."""
        prompt_text = req.prompt
        session_id = req.session_id or "default"

        history = _session_manager.get_history(session_id)

        if req.interrupted_context and isinstance(req.interrupted_context, dict):
            prev_user = req.interrupted_context.get("previousUserPrompt", "").strip()
            prev_ai = req.interrupted_context.get("previousAssistantText", "").strip()
            if prev_user or prev_ai:
                prompt_text = (
                    f"The user interrupted an answer that was already spoken aloud.\n\n"
                    f"Previous user request:\n{prev_user}\n\n"
                    f"Portion of the assistant answer that was already spoken:\n{prev_ai}\n\n"
                    f"The user's new interruption/request:\n{req.prompt}\n\n"
                    f"Continue the conversation naturally.\n"
                    f"Do not repeat the portion of the answer that has already been spoken.\n"
                    f"Address the user's new request and continue from the existing context.\n"
                    f"Return only the new spoken content that should be played after the already-spoken answer.\n"
                    f"Do not use markdown, bullets, headings, or meta commentary."
                )

        response_text = _llm.generate_response(prompt_text, history=history)

        # Record turns in in-memory session history
        _session_manager.add_turn(session_id, "user", req.prompt)
        if response_text and not response_text.startswith("[Gemini Error"):
            _session_manager.add_turn(session_id, "model", response_text)

        return JSONResponse(
            {"response": response_text},
            headers={"X-Session-ID": session_id}
        )

    @app.post("/session/reset")
    async def reset_session(req: SessionResetRequest):
        """Clear conversation history for a given session."""
        _session_manager.reset_session(req.session_id)
        return JSONResponse({"status": "ok", "session_id": req.session_id})

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

    import os
    if os.path.exists("web"):
        from fastapi.staticfiles import StaticFiles

        class SafeStaticFiles(StaticFiles):
            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return
                await super().__call__(scope, receive, send)

        app.mount("/", SafeStaticFiles(directory="web", html=True), name="static")

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
