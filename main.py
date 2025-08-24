#!/usr/bin/env python3
"""
Main Server Entry Point

Runs the inference engine server with proper logging and error handling.
"""

import logging
import sys
from pathlib import Path
import grpc
from concurrent import futures
import inference_pb2_grpc

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.grpc_service import InferenceService
from core.config import InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference_engine.log"),
    ],
)
logger = logging.getLogger(__name__)


def setup_environment(config: InferenceConfig):
    """Setup environment and check prerequisites"""
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("Models directory not found. Creating it...")
        models_dir.mkdir(exist_ok=True)
        logger.info("Please place your ONNX model in models/gpt2.onnx")

    # Check if model file exists
    model_path = Path(config.batcher.model_path)
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        logger.info("Please download or place your ONNX model in models/gpt2.onnx")


def main():
    """Main server entry point"""
    try:
        logger.info("Starting Inference Engine Server...")

        # Load configuration (could be extended to parse env/args)
        config = InferenceConfig()

        # Setup environment
        setup_environment(config)

        # Initialize the inference service
        service = InferenceService(config=config)
        logger.info("Inference service initialized successfully")

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=config.server.max_workers)
        )
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
        bind_addr = f"{config.server.host}:{config.server.port}"
        server.add_insecure_port(bind_addr)
        server.start()
        logger.info(f"gRPC server started on {bind_addr}")

        # Keep the server running
        logger.info("Press Ctrl+C to stop the server")
        try:
            while True:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
            service.batcher.shutdown()
            server.stop(0)
            logger.info("Server shutdown complete")

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
