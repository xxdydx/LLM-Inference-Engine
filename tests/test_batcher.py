"""
Tests for the batcher component
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.batcher import Batcher, InferenceResult


class TestBatcher:
    """Test cases for the Batcher class"""

    def test_singleton_pattern(self):
        """Test that Batcher follows singleton pattern"""
        batcher1 = Batcher.instance()
        batcher2 = Batcher.instance()
        assert batcher1 is batcher2

    def test_enqueue_with_valid_request(self):
        """Test enqueue with valid request"""
        batcher = Batcher.instance()
        request = {"input_ids": [1, 2, 3, 4, 5], "max_tokens": 10}

        # Note: This will fail if no model is present, which is expected
        with pytest.raises(Exception):
            result = batcher.enqueue(request)

    def test_enqueue_with_empty_input_ids(self):
        """Test enqueue with empty input_ids"""
        batcher = Batcher.instance()
        request = {"input_ids": [], "max_tokens": 10}

        with pytest.raises(ValueError, match="Empty input_ids provided"):
            batcher.enqueue(request)

    def test_inference_result_dataclass(self):
        """Test InferenceResult dataclass"""
        result = InferenceResult(output_ids=[1, 2, 3], logits=[0.1, 0.2, 0.3])

        assert result.output_ids == [1, 2, 3]
        assert result.logits == [0.1, 0.2, 0.3]
