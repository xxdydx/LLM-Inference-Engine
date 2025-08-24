"""
Token Generation Module

Handles token generation logic with KV caching and batch processing.
"""

import numpy as np
import time
import logging
from typing import List, Optional, Tuple, Dict
from collections import deque
from .beam_search import BeamSearchDecoder, BeamHypothesis

logger = logging.getLogger(__name__)


class TokenGenerator:
    """Handles token generation with KV caching and performance optimization"""

    def __init__(self, inference_engine):
        """
        Initialize token generator

        Args:
            inference_engine: The ONNX inference engine instance
        """
        self.engine = inference_engine
        self.inference_times = deque(maxlen=100)

    def _get_cache_key(self, sequence_id: int, position: int) -> str:
        """Generate cache key for KV cache"""
        return f"seq_{sequence_id}_pos_{position}"

    def run_inference(
        self,
        input_tensor: np.ndarray,
        sequence_id: int = 0,
        position: int = 0,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Run inference with KV caching support

        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len)
            sequence_id: ID of the sequence for KV caching
            position: Current position in the sequence for KV caching
            use_cache: Whether to use KV caching

        Returns:
            Logits tensor
        """
        try:
            start_time = time.time()

            # Prepare input feed
            input_feed = {"input_ids": input_tensor}

            # Add past_key_values to input if we have cached values and model supports it
            cached_present_kv = None
            if (
                use_cache
                and self.engine.cache_enabled
                and self.engine.model_analysis["supports_kv_cache"]
                and position > 0
            ):
                cache_key = self._get_cache_key(sequence_id, position - 1)
                cached_present_kv = self.engine.kv_cache.get(cache_key)

                if cached_present_kv is not None:
                    # Add past_key_values to input feed
                    for input_name in self.engine.model_analysis[
                        "past_key_values_inputs"
                    ]:
                        if input_name in cached_present_kv:
                            input_feed[input_name] = cached_present_kv[input_name]
                    logger.debug(
                        f"Using cached KV for sequence {sequence_id} at position {position}"
                    )

            # Determine output names
            output_names = ["logits"]
            if (
                use_cache
                and self.engine.cache_enabled
                and self.engine.model_analysis["supports_kv_cache"]
            ):
                output_names.extend(
                    self.engine.model_analysis["present_key_values_outputs"]
                )

            # Run inference
            outputs = self.engine.session.run(
                output_names=output_names, input_feed=input_feed
            )

            # Extract logits (always first output)
            logits = outputs[0]

            # Update cache with present_key_values if available
            if (
                use_cache
                and self.engine.cache_enabled
                and self.engine.model_analysis["supports_kv_cache"]
                and len(outputs) > 1
            ):
                present_key_values = {}
                for i, output_name in enumerate(
                    self.engine.model_analysis["present_key_values_outputs"]
                ):
                    if i + 1 < len(outputs):  # +1 because logits is first output
                        present_key_values[output_name] = outputs[i + 1]

                if present_key_values:
                    cache_key = self._get_cache_key(sequence_id, position)
                    self.engine.kv_cache.put(cache_key, present_key_values)
                    logger.debug(
                        f"Updated KV cache for seq {sequence_id} pos {position}"
                    )

            # Track performance
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            logger.debug(
                f"Batch inference completed: {input_tensor.shape[0]} requests in {inference_time:.3f}s"
            )

            return logits

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def generate_tokens(
        self,
        input_ids_list: List[List[int]],
        max_lengths: List[int],
        beam_size: int = 1,
        length_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Generate tokens using greedy decoding or beam search with KV caching

        Args:
            input_ids_list: List of input token ID sequences
            max_lengths: List of maximum lengths for each sequence
            beam_size: Number of beams (1 = greedy, >1 = beam search)
            length_penalty: Length penalty for beam search normalization
            eos_token_id: If provided, stop when this token is generated

        Returns:
            List of tuples (output_ids, last_logits) for each request
        """
        batch_size = len(input_ids_list)
        if batch_size == 0:
            return []

        # Initialize output sequences
        output_sequences = [ids.copy() for ids in input_ids_list]
        finished_sequences = [False] * batch_size

        # Find the maximum sequence length to process
        max_seq_len = max(max_lengths)

        if beam_size > 1:
            # Handle beam search separately for each input sequence
            results = []
            for i, (input_seq, max_len) in enumerate(zip(input_ids_list, max_lengths)):
                # Create beam decoder for this sequence
                beam_decoder = BeamSearchDecoder(
                    beam_size=beam_size,
                    max_length=max_len,
                    length_penalty=length_penalty,
                    eos_token_id=eos_token_id,
                )

                # Create a wrapper function for model inference with beam sequence ID
                def model_inference_fn(tokens, beam_sequence_id):
                    input_tensor = np.array([tokens], dtype=np.int64)
                    logits = self.run_inference(
                        input_tensor,
                        sequence_id=beam_sequence_id,
                        position=len(tokens) - 1,
                    )
                    return logits[0, -1, :]

                # Run beam search
                beam_hypotheses: List[BeamHypothesis] = beam_decoder.search(
                    initial_tokens=input_seq,
                    model_inference_fn=model_inference_fn,
                    vocab_size=50000,
                    sequence_id_base=i,
                )

                # Return the best hypothesis
                if beam_hypotheses:
                    best_hypo: BeamHypothesis = beam_hypotheses[0]
                    results.append((best_hypo.tokens, []))
                else:
                    results.append((input_seq, []))

            return results

        # Process each token position with KV caching
        for pos in range(max_seq_len):
            if all(finished_sequences):  # Check if all sequences are finished
                break

            # Prepare batch input tensor
            batch_input = []
            active_indices = []

            for i in range(batch_size):
                if (
                    not finished_sequences[i]
                    and len(output_sequences[i]) < max_lengths[i]
                ):
                    # Pad shorter sequences to current position
                    while len(output_sequences[i]) <= pos:
                        output_sequences[i].append(0)  # pad with 0
                    batch_input.append(output_sequences[i])
                    active_indices.append(i)
                else:
                    # This sequence is finished, add padding
                    padded_seq = output_sequences[i] + [0] * (
                        pos + 1 - len(output_sequences[i])
                    )
                    batch_input.append(padded_seq)

            if not active_indices:
                break

            # Run batched inference with KV caching
            input_tensor = np.array(batch_input, dtype=np.int64)

            if (
                self.engine.cache_enabled
                and active_indices
                and self.engine.model_analysis["supports_kv_cache"]
            ):
                # Use KV caching for the first active sequence
                first_active = active_indices[0]
                logits = self.run_inference(
                    input_tensor, sequence_id=first_active, position=pos, use_cache=True
                )
            else:
                logits = self.run_inference(input_tensor, use_cache=False)

            # Process results for active sequences
            for batch_idx, seq_idx in enumerate(active_indices):
                if len(output_sequences[seq_idx]) <= pos:
                    # Get the next token for this sequence (greedy only - beam search handled above)
                    last_logits = logits[batch_idx, pos, :]
                    next_token = int(np.argmax(last_logits))
                    output_sequences[seq_idx].append(next_token)

                    # Check for EOS token
                    if eos_token_id is not None and next_token == eos_token_id:
                        finished_sequences[seq_idx] = True

        # Return results with last logits for each sequence
        results = []
        for i, seq in enumerate(output_sequences):
            # Get the last logits for this sequence
            if len(seq) > 0:
                # Run single inference to get last logits
                single_input = np.array([seq], dtype=np.int64)
                if (
                    self.engine.cache_enabled
                    and self.engine.model_analysis["supports_kv_cache"]
                ):
                    single_logits = self.run_inference(
                        single_input,
                        sequence_id=i,
                        position=len(seq) - 1,
                        use_cache=True,
                    )
                else:
                    single_logits = self.run_inference(single_input, use_cache=False)
                last_logits = single_logits[0, -1, :].tolist()
            else:
                last_logits = []
            results.append((seq, last_logits))

        return results

    def get_performance_metrics(self) -> Dict:
        """Get token generation performance metrics"""
        if not self.inference_times:
            return {}

        avg_inference_time = np.mean(self.inference_times)
        min_inference_time = np.min(self.inference_times)
        max_inference_time = np.max(self.inference_times)

        return {
            "avg_inference_time_ms": avg_inference_time * 1000,
            "min_inference_time_ms": min_inference_time * 1000,
            "max_inference_time_ms": max_inference_time * 1000,
            "total_inferences": len(self.inference_times),
            "throughput_req_per_sec": (
                1.0 / avg_inference_time if avg_inference_time > 0 else 0
            ),
        }
