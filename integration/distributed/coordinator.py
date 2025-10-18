"""Platform-agnostic distributed training coordinator."""

import os
import torch
import torch.distributed as dist
from typing import Optional


class DistributedCoordinator:
    """
    Platform-agnostic distributed training coordinator.
    Auto-detects and configures distributed training environment.
    """

    def __init__(self):
        """Initialize distributed coordinator."""
        # Get environment variables
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", os.getenv("SLURM_PROCID", "0")))
        self.local_rank = int(os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0")))
        self.node_rank = int(os.getenv("NODE_RANK", "0"))

        self.is_distributed = self.world_size > 1
        self.is_main_process = self.rank == 0
        self.device = None
        self.backend = None

    def setup_distributed(self) -> str:
        """
        Setup distributed training environment.
        Auto-detects backend (NCCL for NVIDIA, RCCL for AMD, Gloo for CPU).

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        if not self.is_distributed:
            # Single process training
            if torch.cuda.is_available():
                self.device = f"cuda:{self.local_rank}"
                torch.cuda.set_device(self.device)
            else:
                self.device = "cpu"

            print(f"Running in single-process mode on {self.device}")
            return self.device

        # Distributed training setup
        print(f"Setting up distributed training:")
        print(f"  World size: {self.world_size}")
        print(f"  Rank: {self.rank}")
        print(f"  Local rank: {self.local_rank}")
        print(f"  Node rank: {self.node_rank}")

        # Determine backend
        if torch.cuda.is_available():
            # Check if ROCm (AMD)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                self.backend = "nccl"  # RCCL for AMD (uses NCCL API)
                self.device = f"cuda:{self.local_rank}"
                print(f"  Detected AMD ROCm, using RCCL backend")
            else:
                # NVIDIA
                self.backend = "nccl"
                self.device = f"cuda:{self.local_rank}"
                print(f"  Detected NVIDIA CUDA, using NCCL backend")

            torch.cuda.set_device(self.local_rank)
        else:
            # CPU
            self.backend = "gloo"
            self.device = "cpu"
            print(f"  No GPU detected, using Gloo backend")

        # Initialize process group if not already initialized
        if not dist.is_initialized():
            # Get master address and port
            master_addr = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "29500")

            print(f"  Master: {master_addr}:{master_port}")
            print(f"  Backend: {self.backend}")

            # Initialize
            dist.init_process_group(
                backend=self.backend,
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=self.world_size,
                rank=self.rank
            )

            print(f"✅ Distributed process group initialized")

        return self.device

    def barrier(self) -> None:
        """Synchronization barrier across all processes."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def print_once(self, *args, **kwargs) -> None:
        """Print only from main process."""
        if self.is_main_process:
            print(*args, **kwargs)

    def cleanup(self) -> None:
        """Cleanup distributed training."""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Distributed process group destroyed")

    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.world_size

    def get_rank(self) -> int:
        """Get global rank."""
        return self.rank

    def get_local_rank(self) -> int:
        """Get local rank on current node."""
        return self.local_rank

    def is_master(self) -> bool:
        """Check if this is the master process."""
        return self.is_main_process

    def __enter__(self):
        """Context manager entry."""
        self.setup_distributed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
