"""Benchmark LETKF core and eigen solver operations."""

from __future__ import annotations

import itertools
import time

import pandas as pd
import torch

from pyletkf.core.letkf.ops import torch_core

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

BATCH_SIZES = (256, 1024, 4096, 8192)
ENSEMBLE_SIZES = (16, 32, 64)


def _is_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_callable(callable_fn, *, sync_device: torch.device, repeat: int = 3, warmup: int = 1):
    try:
        for _ in range(max(0, warmup)):
            callable_fn()
        _sync_if_cuda(sync_device)
        start = time.perf_counter()
        for _ in range(max(1, repeat)):
            callable_fn()
        _sync_if_cuda(sync_device)
        elapsed = time.perf_counter() - start
        return elapsed / max(1, repeat)
    except NotImplementedError:
        return None
    except RuntimeError as exc:
        if _is_oom(exc):
            if sync_device.type == "cuda":
                torch.cuda.empty_cache()
            return None
        raise

def _make_covariance(batch: int, ens: int, dtype: torch.dtype, device: torch.device):
    mat = torch.randn(batch, ens, ens, dtype=dtype, device=device)
    cov = torch.matmul(mat, mat.transpose(-1, -2))
    eye = torch.eye(ens, dtype=dtype, device=device).expand(batch, -1, -1)
    return cov + eye

def _make_letkf_inputs(batch: int, ens: int, dtype: torch.dtype, device: torch.device):
    dep = torch.randn(batch, ens, dtype=dtype, device=device)
    hdx = torch.randn(batch, ens, ens, dtype=dtype, device=device)
    rdiag = torch.rand(batch, ens, dtype=dtype, device=device) + 1.0
    mask = torch.rand(batch, ens, device=device) > 0.1
    return dep, hdx, rdiag, mask

def benchmark_solver(batch_sizes=BATCH_SIZES,
                     ensemble_sizes=ENSEMBLE_SIZES,
                     device=torch.device("cuda:0"),
                     repeat: int = 3,
                     warmup: int = 1):
    """
        Benchmark the eigen solver (non-chunked).
    """
    device = torch.device(device)
    res = []

    for dtype_name, dtype in DTYPE_MAP.items():
        for batch, ens in itertools.product(batch_sizes, ensemble_sizes):
            matrix = None
            try:
                matrix = _make_covariance(batch, ens, dtype, device)
            except RuntimeError as exc:
                if _is_oom(exc):
                    matrix = None
                else:
                    raise

            if matrix is None:
                elapsed = None
            else:
                def call():
                    torch_core.chunked_eigen_solver(matrix,
                                                    chunk_size=batch,
                                                    device=device)
                elapsed = _benchmark_callable(call, sync_device=device, repeat=repeat, warmup=warmup)

            res.append([dtype_name, batch, ens, elapsed])

            if matrix is not None:
                del matrix
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(res, columns=["dtype", "batch", "ens", "time"])
    df["matrix_elems"] = df["batch"] * (df["ens"] ** 2)
    return df

def benchmark_solver_chunked(batch_sizes=BATCH_SIZES,
                             ensemble_sizes=ENSEMBLE_SIZES,
                             chunk_sizes=(512, 2048, 8192),
                             device=torch.device("cuda:0"),
                             repeat: int = 3,
                             warmup: int = 1):
    """
        Benchmark the chunked eigen solver for different chunk sizes.
    """
    device = torch.device(device)
    chunk_sizes = tuple(int(cs) for cs in chunk_sizes if int(cs) > 0)
    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain at least one positive integer")

    res = []

    for dtype_name, dtype in DTYPE_MAP.items():
        for batch, ens in itertools.product(batch_sizes, ensemble_sizes):
            try:
                matrix = _make_covariance(batch, ens, dtype, device)
            except RuntimeError as exc:
                if _is_oom(exc):
                    matrix = None
                else:
                    raise

            for chunk_size in chunk_sizes:
                if matrix is None:
                    elapsed = None
                else:
                    def call():
                        torch_core.chunked_eigen_solver(matrix,
                                                        chunk_size=chunk_size,
                                                        device=device)
                    elapsed = _benchmark_callable(call, sync_device=device, repeat=repeat, warmup=warmup)
                res.append([dtype_name, batch, ens, chunk_size, elapsed])

            if matrix is not None:
                del matrix
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(res, columns=["dtype", "batch", "ens", "chunk_size", "time"])
    df["matrix_elems"] = df["batch"] * (df["ens"] ** 2)
    return df

def _letkf_call_factory(dep, hdx, rdiag, mask, *, chunked: bool, chunk_size: int | None, device: torch.device):
    def _call():
        dep_in = dep.clone()
        hdx_in = hdx.clone()
        rdiag_in = rdiag.clone()
        mask_in = mask.clone()
        if chunked:
            torch_core.chunked_letkf_core(dep_in,
                                          hdx_in,
                                          rdiag_in,
                                          mask_in,
                                          chunk_size=chunk_size,
                                          device=device)
        else:
            torch_core.letkf_core(dep_in,
                                  hdx_in,
                                  rdiag_in,
                                  mask_in)

    return _call

def benchmark_letkf_core(batch_sizes=BATCH_SIZES,
                         ensemble_sizes=ENSEMBLE_SIZES,
                         device=torch.device("cuda:0"),
                         repeat: int = 3,
                         warmup: int = 1):
    """
        Benchmark the base LETKF core.
    """
    device = torch.device(device)
    res = []

    for dtype_name, dtype in DTYPE_MAP.items():
        for batch, ens in itertools.product(batch_sizes, ensemble_sizes):
            try:
                dep, hdx, rdiag, mask = _make_letkf_inputs(batch, ens, dtype, device)
            except RuntimeError as exc:
                if _is_oom(exc):
                    dep = hdx = rdiag = mask = None
                else:
                    raise

            if dep is None:
                elapsed = None
            else:
                call = _letkf_call_factory(dep, hdx, rdiag, mask,
                                           chunked=False,
                                           chunk_size=None,
                                           device=device)
                elapsed = _benchmark_callable(call, sync_device=device, repeat=repeat, warmup=warmup)

            res.append([dtype_name, batch, ens, elapsed])

            if dep is not None:
                del dep, hdx, rdiag, mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(res, columns=["dtype", "batch", "ens", "time"])
    df["matrix_elems"] = df["batch"] * (df["ens"] ** 2)
    return df

def benchmark_letkf_core_chunked(batch_sizes=BATCH_SIZES,
                                 ensemble_sizes=ENSEMBLE_SIZES,
                                 chunk_sizes=(256, 1024, 4096),
                                 device=torch.device("cuda:0"),
                                 repeat: int = 3,
                                 warmup: int = 1):
    """
        Benchmark chunked LETKF core for several chunk sizes.
    """
    device = torch.device(device)
    chunk_sizes = tuple(int(cs) for cs in chunk_sizes if int(cs) > 0)
    if not chunk_sizes:
        raise ValueError("chunk_sizes must contain at least one positive integer")

    res = []

    for dtype_name, dtype in DTYPE_MAP.items():
        for batch, ens in itertools.product(batch_sizes, ensemble_sizes):
            try:
                dep, hdx, rdiag, mask = _make_letkf_inputs(batch, ens, dtype, device)
            except RuntimeError as exc:
                if _is_oom(exc):
                    dep = hdx = rdiag = mask = None
                else:
                    raise

            for chunk_size in chunk_sizes:
                if dep is None:
                    elapsed = None
                else:
                    call = _letkf_call_factory(dep, hdx, rdiag, mask,
                                               chunked=True,
                                               chunk_size=chunk_size,
                                               device=device)
                    elapsed = _benchmark_callable(call, sync_device=device, repeat=repeat, warmup=warmup)
                res.append([dtype_name, batch, ens, chunk_size, elapsed])

            if dep is not None:
                del dep, hdx, rdiag, mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    df = pd.DataFrame(res, columns=["dtype", "batch", "ens", "chunk_size", "time"])
    df["matrix_elems"] = df["batch"] * (df["ens"] ** 2)
    return df
