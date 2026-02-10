"""Benchmark Torch vs Triton implementations of model state ops."""

from __future__ import annotations

import itertools
import time
from tqdm.auto import tqdm

import pandas as pd
import torch

from pyletkf.model_state.ops import torch as torch_ops
from pyletkf.model_state.ops import triton as triton_ops

try:
    import holoviews as hv
    import hvplot.pandas  # noqa: F401  ensures hvplot extension registered
    _HAS_HOLOVIEWS = True
except ImportError:  # pragma: no cover - optional dependency
    hv = None
    _HAS_HOLOVIEWS = False

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

ENSEMBLE_SIZES = (2, 10, 50, 100, 500, 1000)
GRID_SIZES = (10, 20, 50, 100, 200, 500)

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
    shape = (SCALE_VAR_COUNT, ensemble, grid, grid, PRESSURE_LEVELS)
    state = torch.rand(shape, dtype=dtype, device=device)
    return state

def _benchmark_op(fn, *args, device="cuda", repeat=3, warmup=1):
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
        scale_time = _benchmark_op(module.scale_to_letkf, scale_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00, device=device)
        letkf_time = _benchmark_op(module.letkf_to_scale, letkf_input, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00, device=device)
        results.append([name, scale_time, letkf_time])
        
    del scale_state
    del letkf_input
    torch.cuda.empty_cache()
    return results

def benchmark(device = torch.device("cuda:7")):
    res = []
    for dtype_name, dtype in tqdm(DTYPE_MAP.items()):
        for ensemble, grid in itertools.product(ENSEMBLE_SIZES, GRID_SIZES):
            for entry in _run_case(dtype_name, dtype, ensemble, grid, device):
                res.append([dtype_name, ensemble, grid] + entry)
    res = pd.DataFrame(res, columns=["dtype", "ens", "grid", "backend", "to_letkf", "to_scale"])
    res["N"] = PRESSURE_LEVELS * res["ens"] * (res["grid"] * res["grid"]) * SCALE_VAR_COUNT
    return res

def plot_res(res):
    if not _HAS_HOLOVIEWS:
        raise RuntimeError("plot_res requires holoviews and hvplot to be installed")
    all_plots = []
    
    for backend in ["torch", "triton"]:
        plots = []
        for dt in res["dtype"].unique():
            r = res[res.dtype==dt]
            p = r[r["backend"]==backend].sort_values(by="N").hvplot(x="N", y="to_scale", label=dt)
            df = r[r["backend"]==backend].sort_values(by="N")
        
            line = df.hvplot(
                x="N", y="to_scale",
                label=dt,
                line_width=2
            )
            points = df.hvplot.scatter(
                x="N", y="to_scale",
                size=60,
                alpha=0.8
            )
            
            p = line * points
            plots.append(p)
    
        plots = hv.Overlay(plots).opts(xlim=(0, res[["to_scale", "N"]].dropna()["N"].max()),
                                       ylim=(0, res.dropna()["to_scale"].max()),
                                       title=backend)
        all_plots.append(plots)
    return hv.Layout(all_plots).cols(1)
