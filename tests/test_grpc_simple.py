#!/usr/bin/env python3
"""
Simple test for gRPC beam search protobuf integration
"""

import logging
import inference_pb2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_protobuf_beam_search_fields():
    """Test that protobuf messages have beam search fields"""

    logger.info("🧪 Testing Protobuf Beam Search Fields")

    # Create a request with all beam search parameters
    request = inference_pb2.PredictRequest()
    request.input_ids.extend([1, 2, 3, 4, 5])
    request.max_tokens = 20
    request.beam_size = 3
    request.length_penalty = 1.2
    request.eos_token_id = 50256

    # Verify all fields are accessible
    logger.info(f"input_ids: {list(request.input_ids)}")
    logger.info(f"max_tokens: {request.max_tokens}")
    logger.info(f"beam_size: {request.beam_size}")
    logger.info(f"length_penalty: {request.length_penalty}")
    logger.info(f"eos_token_id: {request.eos_token_id}")

    assert len(request.input_ids) == 5
    assert request.max_tokens == 20
    assert request.beam_size == 3
    assert abs(request.length_penalty - 1.2) < 0.01
    assert request.eos_token_id == 50256

    logger.info("✅ All beam search fields are accessible!")


def test_default_values():
    """Test default values for beam search parameters"""

    logger.info("🧪 Testing Default Values")

    # Create request with minimal parameters
    request = inference_pb2.PredictRequest()
    request.input_ids.extend([1, 2, 3])
    request.max_tokens = 10
    # Don't set beam search parameters - should use defaults

    logger.info(f"Default beam_size: {request.beam_size}")
    logger.info(f"Default length_penalty: {request.length_penalty}")
    logger.info(f"Default eos_token_id: {request.eos_token_id}")

    # In protobuf, unset int32 defaults to 0, float defaults to 0.0
    assert request.beam_size == 0  # Will be corrected to 1 in service
    assert request.length_penalty == 0.0  # Will be corrected to 1.0 in service
    assert request.eos_token_id == 0  # Will be corrected to None in service

    logger.info("✅ Default values test passed!")


def test_serialization():
    """Test protobuf serialization/deserialization with beam search"""

    logger.info("🧪 Testing Protobuf Serialization")

    # Create request
    original_request = inference_pb2.PredictRequest()
    original_request.input_ids.extend([10, 20, 30])
    original_request.max_tokens = 25
    original_request.beam_size = 5
    original_request.length_penalty = 0.8
    original_request.eos_token_id = 12345

    # Serialize
    serialized = original_request.SerializeToString()
    logger.info(f"Serialized size: {len(serialized)} bytes")

    # Deserialize
    deserialized_request = inference_pb2.PredictRequest()
    deserialized_request.ParseFromString(serialized)

    # Verify all fields match
    assert list(deserialized_request.input_ids) == [10, 20, 30]
    assert deserialized_request.max_tokens == 25
    assert deserialized_request.beam_size == 5
    assert abs(deserialized_request.length_penalty - 0.8) < 0.01
    assert deserialized_request.eos_token_id == 12345

    logger.info("✅ Serialization test passed!")


if __name__ == "__main__":
    print("🚀 Starting Simple gRPC Beam Search Tests\\n")

    try:
        # Test 1: Field accessibility
        test_protobuf_beam_search_fields()
        print()

        # Test 2: Default values
        test_default_values()
        print()

        # Test 3: Serialization
        test_serialization()
        print()

        print("🎉 All simple gRPC tests passed!")
        print("✅ Beam search parameters are properly integrated into gRPC interface!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
