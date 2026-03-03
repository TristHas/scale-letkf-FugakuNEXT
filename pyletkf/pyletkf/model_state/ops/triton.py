"""Triton implementations of SCALE <-> LETKF conversion ops."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SCALE_VAR_COUNT = 11
LETKF_VAR_COUNT = 11
NUM_MOIST = 6

def _normalize_cvap(cvap) -> tuple[float, ...]:
    if isinstance(cvap, (int, float)):
        values = [float(cvap)] * NUM_MOIST
    else:
        if isinstance(cvap, torch.Tensor):
            values = cvap.detach().cpu().flatten().tolist()
        else:
            values = list(cvap)
        if len(values) == 1:
            values = values * NUM_MOIST
    if len(values) != NUM_MOIST:
        raise ValueError(f"Expected {NUM_MOIST} specific heat values, got {len(values)}")
    return tuple(float(v) for v in values)

def _flatten_state(state: torch.Tensor, expected: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if state.ndim < 2:
        raise ValueError("state tensor must have at least 2 dimensions")
    if state.shape[0] != expected:
        raise ValueError(f"Expected {expected} variables, got {state.shape[0]}")
    if not state.is_contiguous():
        state = state.contiguous()
    trailing = state.shape[1:]
    return state.reshape(state.shape[0], -1), trailing

@triton.jit
def _scale_to_letkf_kernel(
        state_ptr,
        out_ptr,
        stride_state_var,
        stride_state_col,
        stride_out_var,
        stride_out_col,
        num_cols,
        rdry,
        cvap0,
        cvap1,
        cvap2,
        cvap3,
        cvap4,
        cvap5,
        rvap,
        cvdry,
        pre00,
        BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    pid = pid.to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    cols = cols.to(tl.int64)
    cols = pid * BLOCK_SIZE + cols

    num_cols_i64 = tl.full([], num_cols, dtype=tl.int64)
    mask = cols < num_cols_i64

    stride_state_var_i64 = tl.full([], stride_state_var, dtype=tl.int64)
    stride_state_col_i64 = tl.full([], stride_state_col, dtype=tl.int64)
    stride_out_var_i64 = tl.full([], stride_out_var, dtype=tl.int64)
    stride_out_col_i64 = tl.full([], stride_out_col, dtype=tl.int64)

    rho = tl.load(state_ptr + 0 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    rhot = tl.load(state_ptr + 1 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)

    mom_x = tl.load(state_ptr + 2 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0) / rho
    mom_y = tl.load(state_ptr + 3 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0) / rho
    mom_z = tl.load(state_ptr + 4 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0) / rho

    qv = tl.load(state_ptr + (5 + 0) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qc = tl.load(state_ptr + (5 + 1) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qr = tl.load(state_ptr + (5 + 2) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qi = tl.load(state_ptr + (5 + 3) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qs = tl.load(state_ptr + (5 + 4) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qg = tl.load(state_ptr + (5 + 5) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)

    qv = tl.where(qv == qv, qv, 0.0)
    qc = tl.where(qc == qc, qc, 0.0)
    qr = tl.where(qr == qr, qr, 0.0)
    qi = tl.where(qi == qi, qi, 0.0)
    qs = tl.where(qs == qs, qs, 0.0)
    qg = tl.where(qg == qg, qg, 0.0)

    moist_sum = qv + qc + qr + qi + qs + qg

    qdry = 1.0 - moist_sum
    rtot = rdry * qdry + rvap * qv
    cv_tot = (
        cvdry * qdry
        + cvap0 * qv
        + cvap1 * qc
        + cvap2 * qr
        + cvap3 * qi
        + cvap4 * qs
        + cvap5 * qg
    )

    base = (rhot * rtot) / pre00
    valid = (base > 0.0) & (rho > 0.0) & (cv_tot > 0.0) & (rtot > 0.0)

    nan = tl.full([BLOCK_SIZE], float("nan"), dtype=rho.dtype)
    gamma = tl.where(valid, (cv_tot + rtot) / cv_tot, nan)
    safe_base = tl.where(valid, base, 1.0)
    log_base = tl.log(safe_base)
    pow_base = tl.exp(gamma * log_base)
    pressure = tl.where(valid, pre00 * pow_base, nan)
    temperature = tl.where(valid, pressure / (rho * rtot), nan)

    tl.store(out_ptr + 0 * stride_out_var_i64 + cols * stride_out_col_i64, mom_x, mask=mask)
    tl.store(out_ptr + 1 * stride_out_var_i64 + cols * stride_out_col_i64, mom_y, mask=mask)
    tl.store(out_ptr + 2 * stride_out_var_i64 + cols * stride_out_col_i64, mom_z, mask=mask)
    tl.store(out_ptr + 3 * stride_out_var_i64 + cols * stride_out_col_i64, temperature, mask=mask)
    tl.store(out_ptr + 4 * stride_out_var_i64 + cols * stride_out_col_i64, pressure, mask=mask)
    tl.store(out_ptr + (5 + 0) * stride_out_var_i64 + cols * stride_out_col_i64, qv, mask=mask)
    tl.store(out_ptr + (5 + 1) * stride_out_var_i64 + cols * stride_out_col_i64, qc, mask=mask)
    tl.store(out_ptr + (5 + 2) * stride_out_var_i64 + cols * stride_out_col_i64, qr, mask=mask)
    tl.store(out_ptr + (5 + 3) * stride_out_var_i64 + cols * stride_out_col_i64, qi, mask=mask)
    tl.store(out_ptr + (5 + 4) * stride_out_var_i64 + cols * stride_out_col_i64, qs, mask=mask)
    tl.store(out_ptr + (5 + 5) * stride_out_var_i64 + cols * stride_out_col_i64, qg, mask=mask)


@triton.jit
def _letkf_to_scale_kernel(
    state_ptr,
    out_ptr,
    stride_state_var,
    stride_state_col,
    stride_out_var,
    stride_out_col,
    num_cols,
    rdry,
    cvap0,
    cvap1,
    cvap2,
    cvap3,
    cvap4,
    cvap5,
    rvap,
    cvdry,
    pre00,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid = pid.to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    cols = cols.to(tl.int64)
    cols = pid * BLOCK_SIZE + cols

    num_cols_i64 = tl.full([], num_cols, dtype=tl.int64)
    mask = cols < num_cols_i64

    stride_state_var_i64 = tl.full([], stride_state_var, dtype=tl.int64)
    stride_state_col_i64 = tl.full([], stride_state_col, dtype=tl.int64)
    stride_out_var_i64 = tl.full([], stride_out_var, dtype=tl.int64)
    stride_out_col_i64 = tl.full([], stride_out_col, dtype=tl.int64)

    mom_x = tl.load(state_ptr + 0 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    mom_y = tl.load(state_ptr + 1 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    mom_z = tl.load(state_ptr + 2 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    temperature = tl.load(state_ptr + 3 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    pressure = tl.load(state_ptr + 4 * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)

    qv = tl.load(state_ptr + (5 + 0) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qc = tl.load(state_ptr + (5 + 1) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qr = tl.load(state_ptr + (5 + 2) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qi = tl.load(state_ptr + (5 + 3) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qs = tl.load(state_ptr + (5 + 4) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)
    qg = tl.load(state_ptr + (5 + 5) * stride_state_var_i64 + cols * stride_state_col_i64, mask=mask, other=0.0)

    qv = tl.where(qv == qv, qv, 0.0)
    qc = tl.where(qc == qc, qc, 0.0)
    qr = tl.where(qr == qr, qr, 0.0)
    qi = tl.where(qi == qi, qi, 0.0)
    qs = tl.where(qs == qs, qs, 0.0)
    qg = tl.where(qg == qg, qg, 0.0)

    moist_sum = qv + qc + qr + qi + qs + qg

    qdry = 1.0 - moist_sum
    rtot = rdry * qdry + rvap * qv
    cv_tot = (
        cvdry * qdry
        + cvap0 * qv
        + cvap1 * qc
        + cvap2 * qr
        + cvap3 * qi
        + cvap4 * qs
        + cvap5 * qg
    )

    valid = (pressure > 0.0) & (temperature > 0.0) & (cv_tot > 0.0) & (rtot > 0.0)

    nan = tl.full([BLOCK_SIZE], float("nan"), dtype=pressure.dtype)
    gamma = tl.where(valid, (cv_tot + rtot) / cv_tot, nan)
    base = tl.where(valid, pressure / pre00, nan)
    inv_gamma = tl.where(valid, 1.0 / gamma, nan)
    safe_base = tl.where(valid, base, 1.0)
    log_base = tl.log(safe_base)
    pow_base = tl.exp(inv_gamma * log_base)
    base = tl.where(valid, pow_base, nan)

    rhot = tl.where(valid, base * pre00 / rtot, nan)
    rho = tl.where(valid, pressure / (rtot * temperature), nan)

    tl.store(out_ptr + 0 * stride_out_var_i64 + cols * stride_out_col_i64, rho, mask=mask)
    tl.store(out_ptr + 1 * stride_out_var_i64 + cols * stride_out_col_i64, rhot, mask=mask)
    tl.store(out_ptr + 2 * stride_out_var_i64 + cols * stride_out_col_i64, mom_x * rho, mask=mask)
    tl.store(out_ptr + 3 * stride_out_var_i64 + cols * stride_out_col_i64, mom_y * rho, mask=mask)
    tl.store(out_ptr + 4 * stride_out_var_i64 + cols * stride_out_col_i64, mom_z * rho, mask=mask)
    tl.store(out_ptr + (5 + 0) * stride_out_var_i64 + cols * stride_out_col_i64, qv, mask=mask)
    tl.store(out_ptr + (5 + 1) * stride_out_var_i64 + cols * stride_out_col_i64, qc, mask=mask)
    tl.store(out_ptr + (5 + 2) * stride_out_var_i64 + cols * stride_out_col_i64, qr, mask=mask)
    tl.store(out_ptr + (5 + 3) * stride_out_var_i64 + cols * stride_out_col_i64, qi, mask=mask)
    tl.store(out_ptr + (5 + 4) * stride_out_var_i64 + cols * stride_out_col_i64, qs, mask=mask)
    tl.store(out_ptr + (5 + 5) * stride_out_var_i64 + cols * stride_out_col_i64, qg, mask=mask)


def _launch_kernel(kernel, state_2d, out_2d, *, rdry, cvap_vals, rvap, cvdry, pre00, block_size=1024):
    num_cols = state_2d.shape[1]
    grid = lambda META: (triton.cdiv(num_cols, META["BLOCK_SIZE"]),)
    kernel[
        grid
    ](
        state_2d,
        out_2d,
        state_2d.stride(0),
        state_2d.stride(1),
        out_2d.stride(0),
        out_2d.stride(1),
        num_cols,
        rdry,
        *cvap_vals,
        rvap,
        cvdry,
        pre00,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def scale_to_letkf(state, rdry, cvap, rvap, cvdry, pre00, *, block_size: int = 1024):
    """Convert a SCALE state tensor to LETKF ordering using Triton."""
    state = state.transpose(0,1).contiguous()
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    device = state.device

    with torch.cuda.device(device):
        state_2d, trailing_shape = _flatten_state(state, SCALE_VAR_COUNT)
        out_2d = torch.empty((LETKF_VAR_COUNT, state_2d.shape[1]), dtype=state.dtype, device=device)

        cvap_vals = _normalize_cvap(cvap)

        _launch_kernel(
            _scale_to_letkf_kernel,
            state_2d,
            out_2d,
            rdry=float(rdry),
            cvap_vals=cvap_vals,
            rvap=float(rvap),
            cvdry=float(cvdry),
            pre00=float(pre00),
            block_size=block_size,
        )

        result = out_2d.reshape((LETKF_VAR_COUNT, *trailing_shape))
        result = result.transpose(0, 1).contiguous()

    return result

def letkf_to_scale(state, rdry, cvap, rvap, cvdry, pre00, *, block_size: int = 1024):
    """
        Convert a LETKF state tensor back to SCALE ordering using Triton.
    """
    state = state.transpose(0,1).contiguous()
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    device = state.device

    with torch.cuda.device(device):
        state_2d, trailing_shape = _flatten_state(state, LETKF_VAR_COUNT)
        out_2d = torch.empty((SCALE_VAR_COUNT, state_2d.shape[1]), dtype=state.dtype, device=device)

        cvap_vals = _normalize_cvap(cvap)

        _launch_kernel(
            _letkf_to_scale_kernel,
            state_2d,
            out_2d,
            rdry=float(rdry),
            cvap_vals=cvap_vals,
            rvap=float(rvap),
            cvdry=float(cvdry),
            pre00=float(pre00),
            block_size=block_size,
        )

        result = out_2d.reshape((SCALE_VAR_COUNT, *trailing_shape))
        result = result.transpose(0, 1).contiguous()

    return result

__all__ = ["scale_to_letkf", "letkf_to_scale"]
