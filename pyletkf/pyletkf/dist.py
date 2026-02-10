import torch.distributed as dist

def init_distributed_from_env(backend: str = "gloo") -> tuple[int, int]:
    """
        Initialize torch.distributed using the standard env variables.
        Returns (rank, world_size).
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29400")
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size