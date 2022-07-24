"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist_torchrun():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    dist.init_process_group(backend="nccl", init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        dev_idx = os.environ['LOCAL_RANK']
        return th.device(f"cuda:{dev_idx}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    rank = os.environ['RANK']
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
