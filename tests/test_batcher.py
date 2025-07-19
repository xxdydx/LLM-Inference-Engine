"""
Tests for the batcher component
"""

import pytest
import sys
import time
import unittest.mock as mock
from pathlib import Path
from concurrent.futures import Future

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.batcher import Batcher, InferenceResult, BatchedRequest


class TestBatcher:
    """Test cases for the Batcher class"""

    @mock.patch("core.batcher.ONNXInfer")
    def test_singleton_pattern(self, mock_onnx_infer):
        """Test that Batcher follows singleton pattern"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher1 = Batcher.instance()
        batcher2 = Batcher.instance()
        assert batcher1 is batcher2

    @mock.patch("core.batcher.ONNXInfer")
    def test_submit_request_with_valid_input(self, mock_onnx_infer):
        """Test submit_request with valid input"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()
        input_ids = [1, 2, 3, 4, 5]
        max_tokens = 10

        future = batcher.submit_request(input_ids, max_tokens)
        assert isinstance(future, Future)

    @mock.patch("core.batcher.ONNXInfer")
    def test_submit_request_with_empty_input_ids(self, mock_onnx_infer):
        """Test submit_request with empty input_ids"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()
        input_ids = []
        max_tokens = 10

        # Should still create a future even with empty input_ids
        future = batcher.submit_request(input_ids, max_tokens)
        assert isinstance(future, Future)

    @mock.patch("core.batcher.ONNXInfer")
    def test_submit_request_increments_total_requests(self, mock_onnx_infer):
        """Test that submit_request increments the total_requests counter"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()
        initial_requests = batcher.total_requests

        # Submit a request
        future = batcher.submit_request([1, 2, 3], 10)

        # Check that total_requests was incremented
        assert batcher.total_requests == initial_requests + 1

    def test_inference_result_dataclass(self):
        """Test InferenceResult dataclass"""
        result = InferenceResult(output_ids=[1, 2, 3], logits=[0.1, 0.2, 0.3])

        assert result.output_ids == [1, 2, 3]
        assert result.logits == [0.1, 0.2, 0.3]

    def test_batched_request_creation(self):
        """Test BatchedRequest creation"""
        future = Future()
        timestamp = time.time()

        batched_request = BatchedRequest(
            input_ids=[1, 2, 3], max_tokens=10, future=future, timestamp=timestamp
        )

        assert batched_request.input_ids == [1, 2, 3]
        assert batched_request.max_tokens == 10
        assert batched_request.future == future
        assert batched_request.timestamp == timestamp

    @mock.patch("core.batcher.ONNXInfer")
    def test_get_metrics_returns_dict(self, mock_onnx_infer):
        """Test that get_metrics returns a dictionary with expected keys"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()
        metrics = batcher.get_metrics()

        assert isinstance(metrics, dict)
        expected_keys = {
            "total_requests",
            "total_batches",
            "avg_batch_size",
            "batch_timeout_ms",
            "max_batch_size",
            "avg_batch_time_ms",
        }
        assert set(metrics.keys()) == expected_keys

    @mock.patch("core.batcher.ONNXInfer")
    def test_get_metrics_initial_values(self, mock_onnx_infer):
        """Test initial values of metrics"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()
        metrics = batcher.get_metrics()

        # Check initial values
        assert metrics["batch_timeout_ms"] == 100
        assert metrics["max_batch_size"] == 10
        assert metrics["avg_batch_size"] == 0.0
        assert metrics["avg_batch_time_ms"] == 0.0

    @mock.patch("core.batcher.ONNXInfer")
    def test_shutdown_method_exists(self, mock_onnx_infer):
        """Test that shutdown method exists and can be called"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()

        # The shutdown method should exist
        assert hasattr(batcher, "shutdown")
        assert callable(batcher.shutdown)

        # Should not raise an exception when called
        # Note: In a real test, we'd want to test the actual shutdown behavior
        # but that's complex due to threading

    @mock.patch("core.batcher.ONNXInfer")
    def test_batcher_initialization_attributes(self, mock_onnx_infer):
        """Test that batcher has all expected attributes after initialization"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()

        # Check core attributes
        assert hasattr(batcher, "engine")
        assert hasattr(batcher, "request_queue")
        assert hasattr(batcher, "batch_timeout_ms")
        assert hasattr(batcher, "max_batch_size")
        assert hasattr(batcher, "_shutdown")
        assert hasattr(batcher, "batch_thread")

        # Check metrics attributes
        assert hasattr(batcher, "total_requests")
        assert hasattr(batcher, "total_batches")
        assert hasattr(batcher, "avg_batch_size")
        assert hasattr(batcher, "total_batch_time")
        assert hasattr(batcher, "batch_times")

    @mock.patch("core.batcher.ONNXInfer")
    def test_batcher_thread_is_daemon(self, mock_onnx_infer):
        """Test that the batch thread is created as a daemon thread"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()

        assert batcher.batch_thread.daemon is True
        assert batcher.batch_thread.is_alive() is True

    @mock.patch("core.batcher.ONNXInfer")
    def test_queue_operations(self, mock_onnx_infer):
        """Test that requests are properly added to the queue"""
        # Mock the ONNXInfer to avoid model loading issues
        mock_onnx_infer.return_value = mock.MagicMock()

        batcher = Batcher.instance()

        # Submit a request
        future = batcher.submit_request([1, 2, 3], 10)

        # Check that the queue has the request
        assert not batcher.request_queue.empty()

        # Get the request from queue
        request = batcher.request_queue.get()
        assert isinstance(request, BatchedRequest)
        assert request.input_ids == [1, 2, 3]
        assert request.max_tokens == 10
        assert request.future == future

    @mock.patch("core.batcher.ONNXInfer")
    def test_process_batch_calls_generate_tokens_batched_and_sets_result(
        self, mock_onnx_infer
    ):
        # Setup mock for the ONNXInfer instance
        mock_engine = mock.MagicMock()
        mock_engine.generate_tokens.return_value = [
            ([1, 2, 3, 4], [0.1, 0.9, 0.0, 0.0]),  # First request result
            ([5, 6, 7, 8], [0.2, 0.8, 0.0, 0.0]),  # Second request result
        ]
        mock_onnx_infer.return_value = mock_engine

        # Re-initialize the batcher to use the mock
        with mock.patch.object(Batcher, "_instance", None):
            batcher = Batcher.instance()
            batcher.engine = mock_engine

            # Create batched requests
            future1 = Future()
            request1 = BatchedRequest(
                input_ids=[1, 2], max_tokens=4, future=future1, timestamp=time.time()
            )
            future2 = Future()
            request2 = BatchedRequest(
                input_ids=[3], max_tokens=4, future=future2, timestamp=time.time()
            )
            batch_requests = [request1, request2]

            # Call the method under test
            batcher._process_batch(batch_requests)

            # Assert that generate_tokens was called once with all requests
            assert mock_engine.generate_tokens.call_count == 1
            call_args = mock_engine.generate_tokens.call_args
            input_ids_list = call_args[0][0]
            max_lengths = call_args[0][1]

            assert input_ids_list == [[1, 2], [3]]
            assert max_lengths == [4, 4]

            # Check the results set on the futures
            result1 = future1.result()
            assert isinstance(result1, InferenceResult)
            assert result1.output_ids == [1, 2, 3, 4]
            assert result1.logits == [0.1, 0.9, 0.0, 0.0]

            result2 = future2.result()
            assert result2.output_ids == [5, 6, 7, 8]
            assert result2.logits == [0.2, 0.8, 0.0, 0.0]

            # Shutdown the batcher to stop the thread
            batcher.shutdown()
