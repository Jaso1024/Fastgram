"""HuggingFace model implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .base import BaseModel, GenerationConfig, GenerationOutput, LogProbOutput


class HuggingFaceModel(BaseModel):
    """Standard HuggingFace model wrapper.

    Handles loading, generation, logprob computation, and saving
    for HuggingFace transformers models.

    Example:
        model = HuggingFaceModel(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            device="cuda",
            dtype="bfloat16",
        )
        model.load()
        output = model.generate(input_ids, config=GenerationConfig(max_new_tokens=256))
    """

    def load(self) -> "HuggingFaceModel":
        """Load the model and tokenizer from HuggingFace."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Resolve device
        if self._device_str is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda", self.local_rank)
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self._device_str)

        # Resolve dtype
        if self._dtype_str is None:
            self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        elif self._dtype_str == "float32":
            self._dtype = torch.float32
        elif self._dtype_str == "float16":
            self._dtype = torch.float16
        elif self._dtype_str == "bfloat16":
            self._dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {self._dtype_str}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        self._model.to(self._device)

        # Wrap with DDP if requested
        if self.use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            device_ids = [self.local_rank] if self._device.type == "cuda" else None
            self._model = DDP(self._model, device_ids=device_ids)

        self._is_loaded = True
        return self

    def generate(
        self,
        input_ids: "torch.Tensor",
        config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """Generate completions using HuggingFace generate()."""
        import torch

        if config is None:
            config = GenerationConfig()

        # Get the underlying model (unwrap DDP if needed)
        m = self._model.module if hasattr(self._model, "module") else self._model

        was_training = m.training
        m.eval()

        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            # Set seed if provided
            if config.seed is not None:
                torch.manual_seed(config.seed)

            output = m.generate(
                input_ids=input_ids.to(self._device),
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 0,
                top_p=config.top_p if config.do_sample else 1.0,
                do_sample=config.do_sample,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

        if was_training:
            m.train()

        num_generated = output.shape[1] - prompt_len

        return GenerationOutput(
            sequences=output,
            stats={
                "num_generated": num_generated,
                "method": "huggingface",
            },
        )

    def compute_logprobs(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor",
        prompt_lens: "torch.Tensor",
    ) -> LogProbOutput:
        """Compute log probabilities for completion tokens."""
        import torch

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        prompt_lens = prompt_lens.to(self._device)

        # Forward pass
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_attn = attention_mask[:, 1:]

        # Compute log probs
        logp_all = torch.log_softmax(shift_logits, dim=-1)
        logp_all = logp_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Create completion mask
        bsz, seq_len = shift_labels.shape
        pos = torch.arange(seq_len, device=self._device).unsqueeze(0).expand(bsz, -1)
        start = (prompt_lens - 1).clamp_min(0).unsqueeze(1)
        completion_mask = (pos >= start) & shift_attn.bool()

        token_count = completion_mask.sum(dim=1)
        token_logp = logp_all * completion_mask

        return LogProbOutput(
            token_logp=token_logp,
            token_mask=completion_mask,
            token_count=token_count,
        )

    def save(self, path: str) -> None:
        """Save model and tokenizer to directory."""
        out_path = Path(path)
        out_path.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP if needed
        m = self._model.module if hasattr(self._model, "module") else self._model
        m.save_pretrained(out_path)
        self._tokenizer.save_pretrained(out_path)

    def create_reference(self) -> "HuggingFaceModel":
        """Create a frozen copy for KL computation.

        Returns a new model instance with frozen weights.
        """
        ref = HuggingFaceModel(
            model_name=self.model_name,
            device=self._device_str,
            dtype=self._dtype_str,
            use_ddp=False,  # Reference doesn't need DDP
            local_rank=self.local_rank,
        )
        ref.load()
        ref.eval()

        # Freeze parameters
        for p in ref._model.parameters():
            p.requires_grad_(False)

        return ref
