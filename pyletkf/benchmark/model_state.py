from __future__ import annotations

import itertools
import time
from tqdm.auto import tqdm

import pandas as pd
import torch

from pyletkf.model_state.ops import torch as torch_ops
from pyletkf.model_state.ops import triton as triton_ops
from pyletkf.model_state.scale import ScaleLetkfConverter

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

RD_RY = 287.04
RV_AP = 461.50
CP_DRY = 1004.64
CP_VAP = 1846.00
CV_DRY = CP_DRY - RD_RY
CV_VAP = CP_VAP - RV_AP
PRE00 = 100000.0

SCALE_VAR_COUNT = 11
NUM_MOIST = 6
PRESSURE_LEVELS = 45

def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()

def _make_scale_state(dtype: torch.dtype, ensemble: int, grid: int, device: torch.device) -> torch.Tensor:
    shape = (ensemble, SCALE_VAR_COUNT, grid, grid, PRESSURE_LEVELS)
    state = torch.rand(shape, dtype=dtype, device=device)
    return state

def benchmark_op(fn, *args, device="cuda", repeat=3, warmup=1):
    try:
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            fn(*args)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        return elapsed / repeat
    except RuntimeError as exc:
        if _is_oom(exc):
            torch.cuda.empty_cache()
            return None
        raise

def _run_case(dtype_name: str, dtype: torch.dtype, ensemble: int, grid: int, device: torch.device):
    header = f"dtype={dtype_name} ensemble={ensemble} grid={grid}"
    try:
        scale_state = _make_scale_state(dtype, ensemble, grid, device)
    except RuntimeError as exc:
        if _is_oom(exc):
            torch.cuda.empty_cache()
            return [["triton", None, None],  ["torch", None, None]]
        raise

    try:
        letkf_input = torch_ops.scale_to_letkf(scale_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)
    except RuntimeError as exc:
        if _is_oom(exc):
            torch.cuda.empty_cache()
            return [["triton", None, None],  ["torch", None, None]]
        raise

    impls = {
        "torch": torch_ops,
        "triton": triton_ops,
    }
    results = []
    
    for name, module in impls.items():
        scale_time = benchmark_op(module.scale_to_letkf, scale_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00, device=device)
        letkf_time = benchmark_op(module.letkf_to_scale, letkf_input, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00, device=device)
        results.append([name, scale_time, letkf_time])
        
    del scale_state
    del letkf_input
    torch.cuda.empty_cache()
    return results

def benchmark(ens_size  = (8, 32, 128, 512, 1024),
              grid_size = (32, 64, 128, 384),
              device = torch.device("cuda:7")):

    res = []
    for dtype_name, dtype in tqdm(DTYPE_MAP.items()):
        for ensemble, grid in itertools.product(ens_size, grid_size):
            for entry in _run_case(dtype_name, dtype, ensemble, grid, device):
                res.append([dtype_name, ensemble, grid] + entry)
    res = pd.DataFrame(res, columns=["dtype", "ens", "grid", "backend", "to_letkf", "to_scale"])
    res["N"] = PRESSURE_LEVELS * res["ens"] * (res["grid"] * res["grid"]) * SCALE_VAR_COUNT
    return res

def _benchmark_chunked_op(converter: ScaleLetkfConverter,
                          fn,
                          state: torch.Tensor | None,
                          sync_device: torch.device,
                          repeat: int = 3,
                          warmup: int = 1):
    """Benchmark a chunked application, returning None if it OOMs."""
    if state is None:
        return None
    device = torch.device(sync_device)
    try:
        for _ in range(warmup):
            converter.chunked_apply(state, fn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            converter.chunked_apply(state, fn)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return (time.perf_counter() - start) / max(1, repeat)
    except RuntimeError as exc:
        if _is_oom(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        raise


def benchmark_chunk(device = torch.device("cuda:7"),
                    chunk_sizes=(5, 10, 50, 100),
                    repeat: int = 3,
                    warmup: int = 1,
                    *,
                    ens: int = 1000,
                    grid=(256, 320)):
    """Benchmark chunked converter timings for a (possibly custom-sized) CPU-resident state."""
    device = torch.device(device)
    chunk_sizes = tuple(int(cs) for cs in chunk_sizes if int(cs) > 0)
    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain at least one positive integer")

    if isinstance(grid, int):
        grid_x = grid_y = int(grid)
    else:
        if len(grid) != 2:
            raise ValueError("grid must be an int or a tuple/list of two ints")
        grid_x, grid_y = (int(grid[0]), int(grid[1]))
    if ens <= 0 or grid_x <= 0 or grid_y <= 0:
        raise ValueError("ens and grid dimensions must be positive integers")

    state_shape = (ens, SCALE_VAR_COUNT, grid_x, grid_y, PRESSURE_LEVELS)
    impls = {
        "torch": torch_ops,
        "triton": triton_ops,
    }
    converters = {int(cs): ScaleLetkfConverter(device=device, chunk_size=int(cs)) for cs in chunk_sizes}
    results = []

    for dtype_name, dtype in tqdm(DTYPE_MAP.items()):
        try:
            scale_state = torch.rand(state_shape, dtype=dtype, device="cpu")
        except RuntimeError as exc:
            if _is_oom(exc):
                scale_state = None
            else:
                raise

        to_letkf_times = {}
        if scale_state is not None:
            for cs, converter in converters.items():
                for backend_name, module in impls.items():
                    key = (cs, backend_name)
                    to_letkf_times[key] = _benchmark_chunked_op(
                        converter,
                        module.scale_to_letkf,
                        scale_state,
                        device,
                        repeat=repeat,
                        warmup=warmup,
                    )

        try:
            letkf_input = None
            if scale_state is not None:
                letkf_input = torch_ops.scale_to_letkf(
                    scale_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00
                )
        except RuntimeError as exc:
            if _is_oom(exc):
                letkf_input = None
            else:
                raise
        finally:
            if scale_state is not None:
                del scale_state

        for cs, converter in converters.items():
            for backend_name, module in impls.items():
                to_scale_time = _benchmark_chunked_op(
                    converter,
                    module.letkf_to_scale,
                    letkf_input,
                    device,
                    repeat=repeat,
                    warmup=warmup,
                )
                results.append([
                    dtype_name,
                    state_shape[0],
                    state_shape[2],
                    state_shape[3],
                    cs,
                    backend_name,
                    to_letkf_times.get((cs, backend_name)),
                    to_scale_time,
                ])

        if letkf_input is not None:
            del letkf_input

    res = pd.DataFrame(results, columns=["dtype", "ens", 
                                         "grid_x", "grid_y", 
                                         "chunk_size", "backend", 
                                         "to_letkf", "to_scale"])
    res["N"] = res["ens"] * res["grid_x"] * res["grid_y"]\
                          * PRESSURE_LEVELS * SCALE_VAR_COUNT
    return res
