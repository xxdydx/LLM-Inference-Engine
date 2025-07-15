#!/usr/bin/env python3
"""
Main Server Entry Point

Runs the inference engine server with proper logging and error handling.
"""

import logging
import sys
import os
from pathlib import Path
import grpc
from concurrent import futures
from services.grpc_service import InferenceService
import inference_pb2_grpc

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.grpc_service import InferenceService

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


def setup_environment():
    """Setup environment and check prerequisites"""
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        logger.warning("Models directory not found. Creating it...")
        models_dir.mkdir(exist_ok=True)
        logger.info("Please place your ONNX model in models/gpt2.onnx")

    # Check if model file exists
    model_path = models_dir / "gpt2.onnx"
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        logger.info("Please download or place your ONNX model in models/gpt2.onnx")


def main():
    """Main server entry point"""
    try:
        logger.info("Starting Inference Engine Server...")

        # Setup environment
        setup_environment()

        # Initialize the inference service
        service = InferenceService()
        logger.info("Inference service initialized successfully")

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
        server.add_insecure_port("[::]:50051")
        server.start()
        logger.info("gRPC server started on port 50051")

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
