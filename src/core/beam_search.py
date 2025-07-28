from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import heapq


@dataclass
class BeamHypothesis:
    """Beam search hypothesis"""

    tokens: List[int]
    log_prob: float
    is_finished: bool = False  # Hit EOS token?

    @property
    def length(self) -> int:
        return len(self.tokens)

    def get_normalised_score(self, length_penalty: float = 1.0) -> float:
        """Length-normalised score for fair comparison"""
        if length_penalty == 0.0:
            return self.log_prob

        return self.log_prob / (self.length**length_penalty)

    def add_token(self, token_id: int, token_log_prob: float) -> "BeamHypothesis":
        """Creates a new hypothesis with the added token"""

        new_tokens = self.tokens + [token_id]
        new_log_prob = self.log_prob + token_log_prob

        return BeamHypothesis(
            tokens=new_tokens,
            log_prob=new_log_prob,
            is_finished=self.is_finished,
        )


class BeamSearchDecoder:
    def __init__(
        self,
        beam_size: int = 4,
        max_length: int = 100,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        early_stopping: bool = True,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.eos_token_id = eos_token_id
        self.early_stopping = early_stopping

    def _get_top_k_tokens(
        self, logits: np.ndarray, k: int, hypo: BeamHypothesis
    ) -> List[tuple]:
        """Get top k tokens from logits with repetition penalty"""

        # Applying repetition penalty
        if self.repetition_penalty != 1.0:
            penalty_mask = np.ones_like(logits)
            for token_id in hypo.tokens:
                if token_id < len(penalty_mask):
                    penalty_factor = 1.0 / self.repetition_penalty
                    penalty_mask[token_id] = penalty_factor

            logits = logits * penalty_mask

        # Normalising logits — for numerical stability, largest value is 0
        logits_stable = logits - np.max(logits)
        probs = np.exp(logits_stable) / np.sum(np.exp(logits_stable))

        # Get top k tokens
        top_k_indices = np.argsort(probs)[-k:][::-1]

        results = []
        for token_id in top_k_indices:
            # Converting to log probability & adding small epsilon to avoid log(0)
            log_prob = np.log(probs[token_id] + 1e-10)
            results.append((int(token_id), float(log_prob)))

        return results

    def search(
        self,
        initial_tokens: List[int],
        model_inference_fn: callable,
        vocab_size: int,
        sequence_id_base: int = 0,
    ) -> List[BeamHypothesis]:
        """Main beam search algorithm"""

        current_beam = [
            BeamHypothesis(tokens=initial_tokens, log_prob=0.0, is_finished=False)
        ]
        finished_hypos = []

        for step in range(self.max_length - len(initial_tokens)):
            if not current_beam:
                break

            all_candidates = []

            for beam_idx, hypo in enumerate(current_beam):
                if hypo.is_finished:
                    finished_hypos.append(hypo)
                    continue

                # get model predicitions for this hypothesis
                # Calculate unique sequence ID for this beam
                beam_sequence_id = sequence_id_base * 1000 + beam_idx
                logits = model_inference_fn(hypo.tokens, beam_sequence_id)

                # get top k tokens for expansion
                top_tokens = self._get_top_k_tokens(
                    logits, k=self.beam_size * 2, hypo=hypo
                )

                for token_id, token_log_prob in top_tokens:
                    new_hypo = hypo.add_token(token_id, token_log_prob)

                    if self.eos_token_id is not None and token_id == self.eos_token_id:
                        new_hypo.is_finished = True

                    score = new_hypo.get_normalised_score(self.length_penalty)
                    heapq.heappush(all_candidates, (-score, new_hypo))

            # Keep only top beam_size candidates using heap
            top_candidates = []
            while all_candidates and len(top_candidates) < self.beam_size:
                _, hypo = heapq.heappop(all_candidates)
                top_candidates.append(hypo)

            current_beam = []
            for hypo in top_candidates:
                if hypo.is_finished:
                    finished_hypos.append(hypo)
                else:
                    current_beam.append(hypo)

            if self.early_stopping and len(finished_hypos) >= self.beam_size:
                break

        # Add remaining hypotheses to finished
        finished_hypos.extend(current_beam)
        finished_hypos.sort(
            key=lambda x: x.get_normalised_score(self.length_penalty), reverse=True
        )

        return finished_hypos
