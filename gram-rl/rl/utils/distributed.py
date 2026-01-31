"""Distributed training utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class DistributedInfo:
    """Information about the distributed training environment."""
    rank: int
    world_size: int
    local_rank: int
    is_main: bool
    backend: str
    device: Any  # torch.device


def get_distributed_info() -> DistributedInfo:
    """Get information about the distributed environment.

    Works with both torchrun and single-process execution.
    """
    import torch

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    return DistributedInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=(rank == 0),
        backend=backend,
        device=device,
    )


def init_distributed(info: Optional[DistributedInfo] = None) -> DistributedInfo:
    """Initialize distributed training if running with multiple processes.

    Args:
        info: Optional pre-computed distributed info

    Returns:
        DistributedInfo with initialized process group if multi-process
    """
    import torch
    import torch.distributed as dist

    if info is None:
        info = get_distributed_info()

    if info.world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=info.backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)

    return info


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: "torch.Tensor") -> "torch.Tensor":
    """All-reduce a tensor by taking the mean across processes.

    No-op if not in distributed mode.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


def all_reduce_sum(tensor: "torch.Tensor") -> "torch.Tensor":
    """All-reduce a tensor by summing across processes.

    No-op if not in distributed mode.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast a Python object from src rank to all ranks.

    No-op if not in distributed mode.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier() -> None:
    """Synchronize all processes.

    No-op if not in distributed mode.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_scalars(values: list[float], device: "torch.device") -> "torch.Tensor":
    """Gather scalar values from all processes.

    Args:
        values: List of scalar values from this process
        device: Device to create tensors on

    Returns:
        Tensor with summed values across all processes
    """
    import torch

    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    return all_reduce_sum(tensor)
