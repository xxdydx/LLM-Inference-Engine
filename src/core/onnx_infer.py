"""
ONNX Inference Engine

Handles model loading and inference using ONNX Runtime.
"""

import onnxruntime as ort
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ONNXInfer:
    """ONNX inference engine for running model inference"""

    def __init__(self, model_path: str):
        """Initialize ONNX inference engine with model path"""
        try:
            self.opts = ort.SessionOptions()
            self.opts.intra_op_num_threads = 1  # single thread for inference
            self.session = ort.InferenceSession(
                model_path, self.opts, providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model from {model_path}: {e}")
            raise

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference on a batch of input token IDs and return logits"""
        try:
            # input_tensor shape: (batch_size, seq_len)
            outputs = self.session.run(
                output_names=["logits"], input_feed={"input_ids": input_tensor}
            )
            logger.debug(f"Batch Inference completed: {input_tensor.shape[0]} requests")

            return outputs[0]

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def generate_tokens(
        self,
        input_ids_list: List[List[int]],
        max_lengths: List[int],
        eos_token_id: Optional[int] = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Greedy decode from the ONNX model
        Args:
            input_ids_list: List of input token ID sequences.
            max_lengths: List of maximum lengths for each sequence.
            eos_token_id: If provided, stop when this token is generated.

        Returns:
            List of tuples (output_ids, last_logits) for each request.
        """

        batch_size = len(input_ids_list)
        if batch_size == 0:
            return []

        # Initialize output sequences
        output_sequences = [ids.copy() for ids in input_ids_list]
        finished_sequences = [False] * batch_size

        # Find the maximum sequence length to process
        max_seq_len = max(max_lengths)

        # Process each token position
        for pos in range(max_seq_len):
            # Check if all sequences are finished
            if all(finished_sequences):
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

            # Run batched inference
            input_tensor = np.array(batch_input, dtype=np.int64)
            logits = self.run(input_tensor)

            # Process results for active sequences
            for batch_idx, seq_idx in enumerate(active_indices):
                if len(output_sequences[seq_idx]) <= pos:
                    # Get the next token for this sequence
                    last_logits = logits[batch_idx, pos, :]
                    next_token = int(np.argmax(last_logits))
                    output_sequences[seq_idx].append(next_token)

                    # Check for EOS token
                    if eos_token_id is not None and next_token == eos_token_id:
                        finished_sequences[seq_idx] = True

        # Return results with last logits for each sequence
        results = []
        for i, seq in enumerate(output_sequences):
            # Get the last logits for this sequence (we'll need to run one more inference)
            if len(seq) > 0:
                # Run single inference to get last logits
                single_input = np.array([seq], dtype=np.int64)
                single_logits = self.run(single_input)
                last_logits = single_logits[0, -1, :].tolist()
            else:
                last_logits = []
            results.append((seq, last_logits))

        return results
