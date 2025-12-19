"""Torch-based radar observation operator for SCALE LETKF dumps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import math

from .load_dumps import (
    _normalize_member,
    _normalize_pe_tag,
    _read_binary_array,
    load_and_convert_rank_members,
    load_obs_raw,
)
from .params import OBS_ID_CONSTANTS
from .calcref import calc_ref_vr, ReflectivityContext

EARTH_RADIUS = 6_371_000.0
ID_RADAR_REF = int(OBS_ID_CONSTANTS["id_radar_ref_obs"])
RD_AIR = 287.04


@dataclass
class RadarObsBatch:
    lon: np.ndarray
    lat: np.ndarray
    lev: np.ndarray
    elm: np.ndarray


def _load_radar_observations(
    dump_dir: Path,
    pe_tag: str,
    member: str,
) -> RadarObsBatch:
    stage_dir = dump_dir / "obsda_after_set_letkf"
    pe_norm = _normalize_pe_tag(pe_tag)
    mem_norm = _normalize_member(member)
    idx = _read_binary_array(stage_dir / f"obsda_idx_{pe_norm}.{mem_norm}.bin", ">i4").reshape(-1)
    idx = np.clip(idx - 1, 0, None)

    raw_records = load_obs_raw(dump_dir, pe_tag=pe_tag, member=member)
    if not raw_records:
        raise FileNotFoundError("No obs_raw records found")
    raw = raw_records[0]["data"]
    lon = np.asarray(raw["lon"], dtype=np.float64)[idx]
    lat = np.asarray(raw["lat"], dtype=np.float64)[idx]
    lev = np.asarray(raw["lev"], dtype=np.float64)[idx]
    elm = np.asarray(raw["elm"], dtype=np.int64)[idx]
    return RadarObsBatch(lon=lon, lat=lat, lev=lev, elm=elm)

def _fractional_index(coord: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hi = torch.bucketize(values, coord)
    hi = torch.clamp(hi, 1, coord.numel() - 1)
    lo = hi - 1
    denom = coord[hi] - coord[lo]
    frac = torch.zeros_like(values)
    mask = denom != 0
    frac[mask] = (values[mask] - coord[lo[mask]]) / denom[mask]
    return lo, frac

def run_radar_operator_torch(
    dump_dir: str | Path,
    pe_tag: str,
    member: str,
    radar_lon: float,
    radar_lat: float,
    radar_alt: float,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute H(x) for radar reflectivity observations on a tile."""

    dump_path = Path(dump_dir)
    state = load_and_convert_rank_members(dump_path, "gues3d", pe_tag)

    obs_batch = _load_radar_observations(dump_path, pe_tag, member)
    radar_mask = obs_batch.elm == ID_RADAR_REF
    lon_obs = obs_batch.lon[radar_mask]
    lat_obs = obs_batch.lat[radar_mask]
    lev_obs = obs_batch.lev[radar_mask]
    if lon_obs.size == 0:
        return np.empty((0, state.sizes["ens"]), dtype=np.float64)

    device = torch.device(device)
    lon_axis = torch.as_tensor(state.coords["lon"].values, dtype=torch.float64, device=device)
    lat_axis = torch.as_tensor(state.coords["lat"].values, dtype=torch.float64, device=device)
    z_axis = torch.as_tensor(state.coords["z"].values, dtype=torch.float64, device=device)
    state_tensor = torch.from_numpy(np.transpose(state.values, (3, 4, 2, 0, 1))).to(device)

    lon_t = torch.as_tensor(lon_obs, dtype=torch.float64, device=device)
    lat_t = torch.as_tensor(lat_obs, dtype=torch.float64, device=device)
    lev_t = torch.as_tensor(lev_obs, dtype=torch.float64, device=device)

    ix0, fx = _fractional_index(lon_axis, lon_t)
    iy0, fy = _fractional_index(lat_axis, lat_t)
    iz0, fz = _fractional_index(z_axis, lev_t)
    nx = lon_axis.numel()
    ny = lat_axis.numel()
    nz = z_axis.numel()
    ix0 = torch.clamp(ix0.long(), 0, nx - 2)
    iy0 = torch.clamp(iy0.long(), 0, ny - 2)
    iz0 = torch.clamp(iz0.long(), 0, nz - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1

    fx = fx.to(device=device).view(1, 1, -1)
    fy = fy.to(device=device).view(1, 1, -1)
    fz = fz.to(device=device).view(1, 1, -1)

    def gather(ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor) -> torch.Tensor:
        return state_tensor[:, :, iz, iy, ix]

    v000 = gather(ix0, iy0, iz0)
    v100 = gather(ix1, iy0, iz0)
    v010 = gather(ix0, iy1, iz0)
    v110 = gather(ix1, iy1, iz0)
    v001 = gather(ix0, iy0, iz1)
    v101 = gather(ix1, iy0, iz1)
    v011 = gather(ix0, iy1, iz1)
    v111 = gather(ix1, iy1, iz1)

    c00 = v000 * (1.0 - fx) + v100 * fx
    c01 = v010 * (1.0 - fx) + v110 * fx
    c10 = v001 * (1.0 - fx) + v101 * fx
    c11 = v011 * (1.0 - fx) + v111 * fx

    c0 = c00 * (1.0 - fy) + c01 * fy
    c1 = c10 * (1.0 - fy) + c11 * fy
    samples = c0 * (1.0 - fz) + c1 * fz  # (nens, nvar, nobs)

    var_map = {name: idx for idx, name in enumerate(state.coords["variable"].values)}

    samples_np = samples.cpu().numpy()
    lon_np = lon_t.cpu().numpy()
    lat_np = lat_t.cpu().numpy()
    lev_np = lev_t.cpu().numpy()
    ctx = ReflectivityContext()
    hx = np.zeros((samples_np.shape[2], samples_np.shape[0]), dtype=np.float64)
    for obs_idx in range(samples_np.shape[2]):
        lon_obs_val = lon_np[obs_idx]
        lat_obs_val = lat_np[obs_idx]
        lev_obs_val = lev_np[obs_idx]
        dlon = math.radians(lon_obs_val - radar_lon)
        dlat = math.radians(lat_obs_val - radar_lat)
        radar_lat_rad = math.radians(radar_lat)
        lat_rad = math.radians(lat_obs_val)
        a = math.sin(dlat / 2.0) ** 2 + math.cos(radar_lat_rad) * math.cos(lat_rad) * math.sin(dlon / 2.0) ** 2
        dist = 2.0 * EARTH_RADIUS * math.asin(min(1.0, math.sqrt(a)))
        az = math.atan2((lon_obs_val - radar_lon) * math.cos(radar_lat_rad), lat_obs_val - radar_lat)
        if az < 0.0:
            az += 2.0 * math.pi
        elev = math.atan2(lev_obs_val - radar_alt, max(dist, 1.0))
        for ens_idx in range(samples_np.shape[0]):
            u_val, v_val, w_val, T_val, P_val, qv_val, qc_val, qr_val, qi_val, qs_val, qg_val = samples_np[ens_idx, :, obs_idx]
            radar_ref, _ = calc_ref_vr(
                qv_val,
                qc_val,
                qr_val,
                qi_val,
                qs_val,
                qg_val,
                u_val,
                v_val,
                w_val,
                T_val,
                P_val,
                math.degrees(az),
                math.degrees(elev),
                ctx,
            )
            if radar_ref > 0.0:
                hx[obs_idx, ens_idx] = 10.0 * math.log10(radar_ref)
            else:
                hx[obs_idx, ens_idx] = ctx.min_radar_ref_dbz + ctx.low_ref_shift
    return hx


__all__ = ["run_radar_operator_torch"]
