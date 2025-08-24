"""
Simple Configuration Module
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BatcherConfig:
    """Configuration for the batcher"""
    model_path: str = "models/gpt2.onnx"
    batch_timeout_ms: int = 1000
    max_batch_size: int = 16
    max_cache_size_mb: int = 512


@dataclass
class ServerConfig:
    """Configuration for the server"""
    host: str = "[::]"
    port: int = 50051
    max_workers: int = 10
    request_timeout_seconds: int = 30


@dataclass
class InferenceConfig:
    """Main configuration"""
    
    def __init__(
        self,
        batcher_config: Optional[BatcherConfig] = None,
        server_config: Optional[ServerConfig] = None,
    ):
        self.batcher = batcher_config or BatcherConfig()
        self.server = server_config or ServerConfig()


def create_config(
    model_path: Optional[str] = None,
    batch_timeout_ms: Optional[int] = None,
    max_batch_size: Optional[int] = None,
) -> InferenceConfig:
    """Create configuration with optional overrides"""
    batcher_config = BatcherConfig()

    if model_path:
        batcher_config.model_path = model_path
    if batch_timeout_ms:
        batcher_config.batch_timeout_ms = batch_timeout_ms
    if max_batch_size:
        batcher_config.max_batch_size = max_batch_size

    return InferenceConfig(batcher_config=batcher_config)