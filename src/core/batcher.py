"""
Inference Batcher

Handles batching of inference requests using singleton pattern.
"""

import threading
import queue
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import Future
import logging
import numpy as np
from core.onnx_infer import ONNXInfer

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of inference operation"""
    output_ids: List[int]
    logits: List[float]


@dataclass
class BatchedRequest:
    """Request with batching information"""

    input_ids: List[int]
    max_tokens: int
    future: Future
    timestamp: float


class Batcher:
    """Dynamic batching for handling inference requests with timer-based batching"""

    _instance = None

    # Singleton pattern - ensure only one instance of the batcher exists
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Batcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, quantization_type=None):
        if not hasattr(self, "initialized"):
            try:
                # Import quantization type
                from core.onnx_infer import QuantizationType

                # Use dynamic quantization by default if not specified
                if quantization_type is None:
                    quantization_type = QuantizationType.DYNAMIC

                self.engine = ONNXInfer("models/gpt2.onnx", quantization_type)
                self.request_queue = queue.Queue()
                self.batch_timeout_ms = 100
                self.max_batch_size = 10

                # Threading
                self._shutdown = False
                self.batch_thread = threading.Thread(
                    target=self._batch_worker, daemon=True
                )  # runs in background, not in main thread
                self.batch_thread.start()

                # Metrics
                self.total_requests = 0
                self.total_batches = 0
                self.avg_batch_size = 0.0
                self.total_batch_time = 0.0
                self.batch_times = []

                self.initialized = True
                logger.info(
                    f"Batcher initialized: timeout={self.batch_timeout_ms}ms, max_batch={self.max_batch_size}"
                )

            except Exception as e:
                logger.error(f"Failed to initialize batcher: {e}")
                raise

    @classmethod
    def instance(cls):
        """Get singleton instance"""
        return cls()

    def submit_request(self, input_ids: List[int], max_tokens: int) -> Future:
        """Submit a request for batching and return a Future"""
        # Create a Future for the result
        future = Future()

        # Create batched request
        batched_request = BatchedRequest(
            input_ids=input_ids,
            max_tokens=max_tokens,
            future=future,
            timestamp=time.time(),
        )

        # Add to queue
        self.request_queue.put(batched_request)

        self.total_requests += 1
        logger.debug(f"Request submitted to batch queue: {len(input_ids)} tokens")
        return future

    def _batch_worker(self):
        """Main batching loop — runs in background thread"""
        logger.info("Starting batch worker thread")

        while not self._shutdown:
            batch_requests = []
            batch_start_time = time.time()

            while len(batch_requests) < self.max_batch_size:
                try:
                    elapsed_time = time.time() - batch_start_time
                    timeout = max(0.001, self.batch_timeout_ms / 1000 - elapsed_time)

                    request = self.request_queue.get(timeout=timeout)
                    batch_requests.append(request)

                except queue.Empty:
                    # timeout reached or no more requests in queue
                    break

            if batch_requests:
                self._process_batch(batch_requests)

        logger.info(f"Batch worker thread exiting")

    def _process_batch(self, batch_requests: List[BatchedRequest]):
        """Process a batch of requests together using true batched inference"""

        try:
            batch_start = time.time()

            # Extract input data for batched processing
            input_ids_list = [req.input_ids for req in batch_requests]
            max_lengths = [req.max_tokens for req in batch_requests]

            # Process all requests together using batched inference
            batch_results = self.engine.generate_tokens(input_ids_list, max_lengths)

            # Set results for each request
            for i, (req, (generated_ids, last_logits)) in enumerate(
                zip(batch_requests, batch_results)
            ):
                try:
                    result = InferenceResult(
                        output_ids=generated_ids, logits=last_logits
                    )
                    req.future.set_result(result)
                except Exception as e:
                    logger.error(f"Failed to set result for request {i}: {e}")
                    req.future.set_exception(e)

            # Update metrics
            batch_time = time.time() - batch_start
            self.total_batches += 1
            self.avg_batch_size = (
                self.avg_batch_size * (self.total_batches - 1) + len(batch_requests)
            ) / self.total_batches

            self.total_batch_time += batch_time
            self.batch_times.append(batch_time)
            if len(self.batch_times) > 100:
                self.batch_times.pop(0)

            logger.info(
                f"Processed batch of {len(batch_requests)} requests in {batch_time:.3f}s using true batching"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all requests in batch
            for req in batch_requests:
                if not req.future.done():
                    req.future.set_exception(e)

    def shutdown(self):
        """Shutdown the batcher gracefully"""
        logger.info("Shutting down DynamicBatcher...")
        self._shutdown = True  # signal batch thread to exit

        if self.batch_thread.is_alive():  # check if thread is still running
            self.batch_thread.join(timeout=5)  # wait for thread to finish

        logger.info("DynamicBatcher shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current batching metrics including quantization performance"""
        avg_batch_time = (
            sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0.0
        )

        # Get quantization metrics from engine
        quantization_metrics = self.engine.get_performance_metrics()

        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": self.avg_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "max_batch_size": self.max_batch_size,
            "avg_batch_time_ms": avg_batch_time * 1000,
            "quantization_type": quantization_metrics.get(
                "quantization_type", "unknown"
            ),
            "avg_inference_time_ms": quantization_metrics.get(
                "avg_inference_time_ms", 0.0
            ),
            "throughput_req_per_sec": quantization_metrics.get(
                "throughput_req_per_sec", 0.0
            ),
            "total_inferences": quantization_metrics.get("total_inferences", 0),
            "kv_cache_enabled": quantization_metrics.get("kv_cache_enabled", False),
            "kv_cache_size": quantization_metrics.get("kv_cache_size", 0),
        }
