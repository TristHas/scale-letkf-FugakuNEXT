import torch
import numpy as np

EARTH_RADIUS = 6_371_000.0

def vr_operator(obs_state, obs_radar):
    device = obs_state.device
    dtype = obs_state.dtype
    los = _compute_los_vectors_torch(
            obs_radar["lon"].values,
            obs_radar["lat"].values,
            obs_radar["lev"].values,
            float(obs_radar.attrs["radar_lon"]),
            float(obs_radar.attrs["radar_lat"]),
            float(obs_radar.attrs["radar_z"]),
            device=device,
            dtype=dtype,
        )
    return compute_vr(obs_state, los)
    
def _ecef_torch(lat_rad: torch.Tensor, lon_rad: torch.Tensor, alt_m: torch.Tensor) -> torch.Tensor:
    r = EARTH_RADIUS + alt_m
    cos_lat = torch.cos(lat_rad)
    return torch.stack(
        (
            r * cos_lat * torch.cos(lon_rad),
            r * cos_lat * torch.sin(lon_rad),
            r * torch.sin(lat_rad),
        ),
        dim=-1,
    )

def _compute_los_vectors_torch(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    lev_m: np.ndarray,
    radar_lon_deg: float,
    radar_lat_deg: float,
    radar_alt_m: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lon = torch.as_tensor(lon_deg, device=device, dtype=dtype).deg2rad()
    lat = torch.as_tensor(lat_deg, device=device, dtype=dtype).deg2rad()
    alt = torch.as_tensor(lev_m, device=device, dtype=dtype)

    radar_lon = torch.as_tensor(radar_lon_deg, device=device, dtype=dtype).deg2rad()
    radar_lat = torch.as_tensor(radar_lat_deg, device=device, dtype=dtype).deg2rad()
    radar_alt = torch.as_tensor(radar_alt_m, device=device, dtype=dtype)

    radar_xyz = _ecef_torch(radar_lat, radar_lon, radar_alt).squeeze(0)
    obs_xyz = _ecef_torch(lat, lon, alt)
    diff = obs_xyz - radar_xyz

    sin_lat0 = torch.sin(radar_lat)
    cos_lat0 = torch.cos(radar_lat)
    sin_lon0 = torch.sin(radar_lon)
    cos_lon0 = torch.cos(radar_lon)

    east = -sin_lon0 * diff[:, 0] + cos_lon0 * diff[:, 1]
    north = (
        -sin_lat0 * cos_lon0 * diff[:, 0]
        - sin_lat0 * sin_lon0 * diff[:, 1]
        + cos_lat0 * diff[:, 2]
    )
    up = (
        cos_lat0 * cos_lon0 * diff[:, 0]
        + cos_lat0 * sin_lon0 * diff[:, 1]
        + sin_lat0 * diff[:, 2]
    )

    los = torch.stack((east, north, up), dim=-1)
    los = torch.nn.functional.normalize(los, dim=-1, eps=1.0e-12)
    return los

def compute_vr(obs_slice: torch.Tensor, los_vectors: torch.Tensor) -> torch.Tensor:
    u = obs_slice[0]
    v = obs_slice[1]
    w = obs_slice[2]
    nx = los_vectors[:, 0].unsqueeze(-1)
    ny = los_vectors[:, 1].unsqueeze(-1)
    nz = los_vectors[:, 2].unsqueeze(-1)
    return u * nx + v * ny + w * nz

