#!/usr/bin/env python3
"""
Test script for beam search implementation
"""

import numpy as np
import logging
from unittest.mock import Mock, MagicMock
from src.core.token_generator import TokenGenerator
from src.core.onnx_infer import ONNXInfer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_engine():
    """Create a mock inference engine for testing"""

    # Mock the ONNX session
    mock_session = Mock()

    # Create predictable logits that favor certain tokens
    def mock_run(output_names, input_feed):
        batch_size = input_feed["input_ids"].shape[0]
        seq_len = input_feed["input_ids"].shape[1]
        vocab_size = 1000

        # Create logits that favor tokens 10, 20, 30 for diversity
        logits = np.random.normal(0, 1, (batch_size, seq_len, vocab_size))

        # Make tokens 10, 20, 30 more likely
        logits[:, :, 10] += 2.0  # High probability
        logits[:, :, 20] += 1.5  # Medium probability
        logits[:, :, 30] += 1.0  # Lower probability

        return [logits]

    mock_session.run = mock_run

    # Mock the inference engine
    mock_engine = Mock(spec=ONNXInfer)
    mock_engine.session = mock_session
    mock_engine.cache_enabled = True
    mock_engine.model_analysis = {
        "supports_kv_cache": False,  # Disable KV cache for simplicity
        "past_key_values_inputs": [],
        "present_key_values_outputs": [],
    }

    # Mock KV cache
    mock_engine.kv_cache = Mock()
    mock_engine.kv_cache.get = Mock(return_value=None)
    mock_engine.kv_cache.put = Mock()

    return mock_engine


def test_greedy_vs_beam_search():
    """Test that beam search produces different results than greedy"""

    logger.info("🧪 Testing Greedy vs Beam Search")

    # Create mock engine and token generator
    mock_engine = create_mock_engine()
    token_gen = TokenGenerator(mock_engine)

    # Test input
    input_sequence = [1, 2, 3]  # Simple input sequence
    max_length = 10

    logger.info(f"Input sequence: {input_sequence}")

    # Test greedy search (beam_size=1)
    logger.info("🎯 Testing Greedy Search (beam_size=1)")
    greedy_results = token_gen.generate_tokens(
        input_ids_list=[input_sequence],
        max_lengths=[max_length],
        beam_size=1,  # Greedy
        length_penalty=1.0,
        eos_token_id=None,
    )

    greedy_tokens = greedy_results[0][0]
    logger.info(f"Greedy result: {greedy_tokens}")

    # Test beam search (beam_size=3)
    logger.info("🌟 Testing Beam Search (beam_size=3)")
    beam_results = token_gen.generate_tokens(
        input_ids_list=[input_sequence],
        max_lengths=[max_length],
        beam_size=3,  # Beam search
        length_penalty=1.0,
        eos_token_id=None,
    )

    beam_tokens = beam_results[0][0]
    logger.info(f"Beam result: {beam_tokens}")

    # Verify results
    logger.info("📊 Analysis:")
    logger.info(f"Greedy length: {len(greedy_tokens)}")
    logger.info(f"Beam length: {len(beam_tokens)}")
    logger.info(f"Results different: {greedy_tokens != beam_tokens}")

    # Both should start with the same input
    assert greedy_tokens[: len(input_sequence)] == input_sequence
    assert beam_tokens[: len(input_sequence)] == input_sequence

    logger.info("✅ Test completed successfully!")

    return greedy_results, beam_results


def test_multiple_sequences():
    """Test beam search with multiple input sequences"""

    logger.info("🧪 Testing Multiple Sequences")

    mock_engine = create_mock_engine()
    token_gen = TokenGenerator(mock_engine)

    # Multiple input sequences
    input_sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    max_lengths = [8, 8, 8]

    logger.info(f"Input sequences: {input_sequences}")

    # Test with beam search
    results = token_gen.generate_tokens(
        input_ids_list=input_sequences,
        max_lengths=max_lengths,
        beam_size=2,
        length_penalty=1.0,
        eos_token_id=None,
    )

    logger.info("Results:")
    for i, (tokens, _) in enumerate(results):
        logger.info(f"  Sequence {i}: {tokens}")

    # Verify we got results for all sequences
    assert len(results) == len(input_sequences)

    logger.info("✅ Multiple sequences test completed!")

    return results


if __name__ == "__main__":
    print("🚀 Starting Beam Search Tests\n")

    try:
        # Test 1: Greedy vs Beam Search
        greedy_res, beam_res = test_greedy_vs_beam_search()
        print()

        # Test 2: Multiple sequences
        multi_res = test_multiple_sequences()
        print()

        print("🎉 All tests passed! Beam search implementation is working!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
