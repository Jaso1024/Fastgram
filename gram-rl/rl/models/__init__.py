"""Model wrappers for RL training.

Provides a unified interface for different model implementations,
abstracting away loading, generation, logprob computation, and saving.

Example:
    from rl.models import HuggingFaceModel

    model = HuggingFaceModel(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda",
    )
    model.load()

    output = model.generate(input_ids, config=GenerationConfig(max_new_tokens=256))
    logprobs = model.compute_logprobs(input_ids, attention_mask, prompt_lens)
"""

from .base import (
    BaseModel,
    GenerationConfig,
    GenerationOutput,
    LogProbOutput,
    pad_sequences,
)
from .huggingface import HuggingFaceModel

__all__ = [
    "BaseModel",
    "GenerationConfig",
    "GenerationOutput",
    "LogProbOutput",
    "HuggingFaceModel",
    "pad_sequences",
]
