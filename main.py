import argparse
import logging
import sys
from config.config import get_config

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("voice-ai-pipeline")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Modular Voice AI Pipeline - Standalone, Client (Tailscale), or Server"
    )
    parser.add_argument(
        "--mode",
        choices=["standalone", "client", "server"],
        default=None,
        help="Operation mode: 'standalone' (local), 'client' (audio on laptop), 'server' (remote host)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="URL of the remote server (e.g., http://100.x.y.z:8000) for client mode",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address to bind to when running in server mode (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number when running in server mode (default: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    """Initialize configuration and run the Voice AI Pipeline or Server."""
    args = parse_args()
    config = get_config()

    # CLI args override environment variables if provided
    if args.mode:
        config.pipeline_mode = args.mode
    if args.server_url:
        config.remote_server_url = args.server_url
    if args.host:
        config.server_host = args.host
    if args.port:
        config.server_port = args.port

    logger.info("Initializing Voice AI Pipeline in '%s' mode...", config.pipeline_mode.upper())

    if config.pipeline_mode == "server":
        # Launch FastAPI server
        import server
        server.main()
        return

    # Client or Standalone mode
    logger.info("Configuration loaded:")
    logger.info("  PIPELINE_MODE: %s", config.pipeline_mode)
    if config.pipeline_mode == "client":
        logger.info("  REMOTE_SERVER_URL: %s", config.remote_server_url)
    else:
        logger.info("  LLM_PROVIDER: %s", config.llm_provider)
        if config.llm_provider.lower() == "gemini":
            logger.info("  GEMINI_MODEL: %s", config.gemini_model)
        else:
            logger.info("  OLLAMA_BASE_URL: %s", config.ollama_base_url)
            logger.info("  OLLAMA_MODEL: %s", config.ollama_model)
        logger.info("  WHISPER_MODEL: %s", config.whisper_model)
        logger.info("  WHISPER_DEVICE: %s", config.whisper_device)
        logger.info("  KOKORO_VOICE: %s", config.kokoro_voice)
    logger.info("  VAD_THRESHOLD: %.2f", config.vad_threshold)

    # Initialize modular voice pipeline
    from src.pipeline.pipeline import VoicePipeline

    pipeline = VoicePipeline(config=config)

    # Execute continuous real-time voice pipeline
    pipeline.run()


if __name__ == "__main__":
    main()
