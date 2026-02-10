import math
import torch

from .utils import linear_reflectivity_method3

EARTH_RADIUS = 6_371_000.0

RD = 287.04
GRAV = 9.80665
PI = math.pi

NOR = 8.0e-2
NOS = 3.0e-2
NOG = 4.0e-2
ROR = 1.0
ROS = 0.1
ROG = 0.4
ROO = 1.28e-3
CD = 0.6
CR_T08 = 130.0
CS_T08 = 4.84
DG_T08 = 0.5
DR_T08 = 0.5
DS_T08 = 0.25
QEPS = 1.0e-20

GAMMA_R = math.gamma(4.0 + DR_T08)
GAMMA_S = math.gamma(4.0 + DS_T08)
GAMMA_G = math.gamma(4.0 + DG_T08)
GRAUPEL_BASE = math.sqrt((4.0 * GRAV * ROG) / (3.0 * CD * ROO))

def filter_vr(obs, gross_error_vr=5., **kwargs):
    """
    """
    return torch.abs(obs["innov"].data) <= gross_error_vr * obs["err"].data

def vr_operator(obs, terminal_velocity=True):
    """
        Radial Velocity Operator
    """
    los = compute_los_vectors(
            obs["lon"].values,
            obs["lat"].values,
            obs["lev"].values,
            float(obs.attrs["radar_lon"]),
            float(obs.attrs["radar_lat"]),
            float(obs.attrs["radar_z"]),
        )
    obs_state = obs["state"].transpose("variable", "obs", "ens")
    winds = obs_state.sel(variable=["U", "V", "W"]).values
    other_state = obs_state.sel(variable=['QR', 'QS', 'QG', 'T', 'P']).values
    return compute_vr(winds, los, other_state, 
                      terminal_velocity=terminal_velocity)

def ecef(lat_rad: torch.Tensor, 
         lon_rad: torch.Tensor, 
         alt_m: torch.Tensor) -> torch.Tensor:
    """
    """
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

def compute_los_vectors(
        lon_deg: torch.Tensor,
        lat_deg: torch.Tensor,
        lev_m: torch.Tensor,
        radar_lon_deg: float,
        radar_lat_deg: float,
        radar_alt_m: float,
    ) -> torch.Tensor:
    """
        Compute Line-of-Sight
    """
    device, dtype = lon_deg.device, lon_deg.dtype
    
    lon = lon_deg.deg2rad()
    lat = lat_deg.deg2rad()
    
    radar_lon = torch.as_tensor(radar_lon_deg, device=device, dtype=dtype).deg2rad()
    radar_lat = torch.as_tensor(radar_lat_deg, device=device, dtype=dtype).deg2rad()
    radar_alt = torch.as_tensor(radar_alt_m, device=device, dtype=dtype)

    radar_xyz = ecef(radar_lat, radar_lon, radar_alt).squeeze(0)
    obs_xyz = ecef(lat, lon, lev_m)
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

def compute_vr(winds: torch.Tensor, 
               los_vectors: torch.Tensor, 
               state: torch.Tensor | None = None,
               terminal_velocity=True) -> torch.Tensor:
    """
    """
    u,v,w = winds
    
    if terminal_velocity and state is not None:
        qr, qs, qg, temp, press = state
        vt = compute_terminal_velocity(
            qr, qs, qg, temp, press,
        )
        w = w - vt

    nx, ny, nz = los_vectors.t().unsqueeze(-1)
    return u * nx + v * ny + w * nz

def compute_terminal_velocity(qr, qs, qg, temp, press,
                             radar_use_melt=False,
                             use_t08_rs2014=False):
    """
    """
    radar_lin, terms = linear_reflectivity_method3(
        qr, qs, qg, temp, press,
        use_melt=radar_use_melt,
        use_t08_rs2014=use_t08_rs2014,
        return_terms=True,
    )
    total_ref = terms["zr"] + terms["zs"] + terms["zg"] + terms["zms"] + terms["zmg"]
    if radar_use_melt:
        z_snow = terms["zs"] + terms["zms"]
        z_graupel = terms["zg"] + terms["zmg"]
    else:
        z_snow = terms["zs"]
        z_graupel = terms["zg"]

    ro = terms["density"].clamp(min=1.0e-12)
    ro_cgs = ro * 1.0e-3
    rofactor = torch.sqrt((ROO / ro_cgs).clamp(min=0.0))

    wr = torch.zeros_like(qr)
    qr_mask = qr > QEPS
    if qr_mask.any():
        lr = torch.pow((PI * ROR * NOR) / (ro_cgs[qr_mask] * qr[qr_mask]), 0.25)
        wr_vals = CR_T08 * GAMMA_R / (
            6.0 * torch.pow(lr * 1.0e2, DR_T08)
        )
        wr[qr_mask] = wr_vals * rofactor[qr_mask]

    ws = torch.zeros_like(qs)
    qs_mask = qs > QEPS
    if qs_mask.any():
        ls = torch.pow((PI * ROS * NOS) / (ro_cgs[qs_mask] * qs[qs_mask]), 0.25)
        ws_vals = CS_T08 * GAMMA_S / (
            6.0 * torch.pow(ls * 1.0e2, DS_T08)
        )
        ws[qs_mask] = ws_vals * rofactor[qs_mask]

    wg = torch.zeros_like(qg)
    qg_mask = qg > QEPS
    if qg_mask.any():
        lg = torch.pow((PI * ROG * NOG) / (ro_cgs[qg_mask] * qg[qg_mask]), 0.25)
        wg_vals = (
            GAMMA_G
            * GRAUPEL_BASE
            / (6.0 * torch.pow(lg * 1.0e2, DG_T08))
        )
        wg[qg_mask] = wg_vals * rofactor[qg_mask]

    numerator = (
        wr * terms["zr"]
        + ws * z_snow
        + wg * z_graupel
    )
    denominator = total_ref
    wt = torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.zeros_like(denominator),
    )
    # zero terminal velocity if no hydrometeors (radar_lin <= 0)
    wt = torch.where(radar_lin > 0.0, wt, torch.zeros_like(wt))
    return wt
