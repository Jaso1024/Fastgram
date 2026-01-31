"""Base model interface for RL training."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None


@dataclass
class GenerationOutput:
    """Output from generation."""
    sequences: Any  # torch.Tensor [B, T] - full sequences including prompt

    # Statistics (optional, model-dependent)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def num_generated(self) -> int:
        """Number of new tokens generated (approximate)."""
        return self.stats.get("num_generated", 0)


@dataclass
class LogProbOutput:
    """Output from logprob computation."""
    token_logp: Any      # torch.Tensor [B, T] - log probs per token
    token_mask: Any      # torch.Tensor [B, T] - mask for completion tokens
    token_count: Any     # torch.Tensor [B] - count of completion tokens


class BaseModel(ABC):
    """Abstract base class for model wrappers.

    Provides a unified interface for:
    - Model loading and device placement
    - Text generation
    - Log probability computation
    - Checkpointing

    Subclass this to implement specific model backends (HuggingFace, vLLM, etc.)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        use_ddp: bool = False,
        local_rank: int = 0,
    ):
        """Initialize model wrapper.

        Args:
            model_name: Model identifier (e.g., HF model name or path)
            device: Device to load model on ("cuda", "cpu", or None for auto)
            dtype: Data type ("float32", "float16", "bfloat16", or None for auto)
            use_ddp: Whether to wrap with DistributedDataParallel
            local_rank: Local rank for DDP
        """
        self.model_name = model_name
        self._device_str = device
        self._dtype_str = dtype
        self.use_ddp = use_ddp
        self.local_rank = local_rank

        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._is_loaded

    @property
    def device(self) -> "torch.device":
        """The device the model is on."""
        if self._device is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._device

    @property
    def dtype(self) -> "torch.dtype":
        """The data type of the model."""
        if self._dtype is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._dtype

    @property
    def tokenizer(self):
        """The tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.tokenizer.vocab_size

    @abstractmethod
    def load(self) -> "BaseModel":
        """Load the model and tokenizer.

        Returns:
            self for chaining
        """
        ...

    @abstractmethod
    def generate(
        self,
        input_ids: "torch.Tensor",
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """Generate completions for input sequences.

        Args:
            input_ids: [B, T] input token IDs
            config: Generation configuration

        Returns:
            GenerationOutput with sequences and stats
        """
        ...

    @abstractmethod
    def compute_logprobs(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        prompt_lens: "torch.Tensor",
    ) -> LogProbOutput:
        """Compute log probabilities for completions.

        Args:
            input_ids: [B, T] full sequences (prompt + completion)
            attention_mask: [B, T] attention mask
            prompt_lens: [B] length of prompt for each sequence

        Returns:
            LogProbOutput with token-level log probabilities
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Directory to save to
        """
        ...

    def parameters(self) -> Iterator:
        """Get trainable parameters."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        m = self._model.module if hasattr(self._model, "module") else self._model
        return m.parameters()

    def train(self) -> "BaseModel":
        """Set model to training mode."""
        if self._model is not None:
            self._model.train()
        return self

    def eval(self) -> "BaseModel":
        """Set model to evaluation mode."""
        if self._model is not None:
            self._model.eval()
        return self

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}({self.model_name!r}, {status})"


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
    prompt_lens: list[int] | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Pad sequences to equal length for batched model input.

    Args:
        sequences: List of token ID sequences
        pad_token_id: Token ID to use for padding
        prompt_lens: Length of prompt portion for each sequence.
                     If None, defaults to 0 for all sequences.

    Returns:
        Tuple of (input_ids, attention_mask, prompt_lens) tensors
    """
    import torch

    bsz = len(sequences)
    if bsz == 0:
        raise ValueError("empty batch")

    if prompt_lens is None:
        prompt_lens = [0] * bsz
    if len(prompt_lens) != bsz:
        raise ValueError("prompt_lens length mismatch")

    max_len = max(len(s) for s in sequences)
    ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attn[i, :len(seq)] = 1

    pl = torch.tensor(prompt_lens, dtype=torch.long)
    return ids, attn, pl
