import numpy as np
import torch

MIN_RADAR_REF_DBZ = 10.0
MIN_RADAR_REF = 10.0 ** (MIN_RADAR_REF_DBZ / 10.0)  # linear threshold = 10
LOW_REF_SHIFT = -5.0
RADAR_USE_MELT = False

def ref_operator(obs_state):
    _, _, _, t, p, _, _, qr, _, qs, qg = obs_state
    radar_lin = linear_reflectivity(qr, qs, qg, t, p,
                                    use_melt=RADAR_USE_MELT)

    hx = torch.full_like(radar_lin, MIN_RADAR_REF_DBZ + LOW_REF_SHIFT)
    min_ref_linear = 10.0 ** (MIN_RADAR_REF_DBZ / 10.0)
    valid = radar_lin >= min_ref_linear
    hx[valid] = 10.0 * torch.log10(radar_lin[valid])
    return hx

def convert_raw_to_dbz(raw):
    """
        Unified interface:
        - If input is a NumPy array, returns a NumPy array.
        - If input is a Torch tensor, returns a Torch tensor.
        Internally dispatches to the torch implementation
    """
    if isinstance(raw, np.ndarray):
        return _convert_raw_to_dbz(torch.from_numpy(raw)).numpy()
    else:
        return _convert_raw_to_dbz(raw)

def _convert_raw_to_dbz(raw_values):
    """
    """
    raw = raw_values.to(torch.float64)
    valid = (raw >= 0.0) & (raw < 1.0e10)
    dbz = torch.full_like(raw, float('nan'))
    # Step 2 — Low reflectivity threshold in *linear units*
    # MIN_RADAR_REF is linear reflectivity threshold (not dBZ)
    min_ref_linear = MIN_RADAR_REF
    low_mask  = valid & (raw < min_ref_linear)
    high_mask = valid & (~low_mask)
    # Step 3 — Allocate outputs for valid values
    # Clear-sky fixed DBZ
    low_values  = torch.full_like(raw, MIN_RADAR_REF_DBZ + LOW_REF_SHIFT)
    # Convert linear reflectivity → dBZ
    high_values = 10.0 * torch.log10(raw.clamp(min=1e-300))  # avoid -inf
    # Step 4 — Combine using differentiable where() (no in-place)
    # Two-step composition to match all conditions
    dbz = torch.where(low_mask,  low_values, dbz)
    dbz = torch.where(high_mask, high_values, dbz)
    return dbz
    
def linear_reflectivity(
        qr: torch.Tensor,
        qs: torch.Tensor,
        qg: torch.Tensor,
        temp: torch.Tensor,
        press: torch.Tensor,
        *,
        use_melt: bool,
        qeps: float = 1.0e-20,
    ) -> torch.Tensor:
    ro = torch.clamp(press / (287.04 * temp), min=1.0e-12)

    if use_melt:
        def _mix_ratio_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ratio = torch.zeros_like(a)
            valid = (a > qeps) & (b > qeps)
            ratio[valid] = torch.minimum(a[valid] / b[valid], b[valid] / a[valid])
            return ratio

        fg = 0.5 * torch.pow(_mix_ratio_ratio(qr, qg), 1.0 / 3.0)
        fs = 0.5 * torch.pow(_mix_ratio_ratio(qr, qs), 1.0 / 3.0)
    else:
        fg = torch.zeros_like(qr)
        fs = torch.zeros_like(qr)

    qrp = torch.clamp((1.0 - fs - fg) * qr, min=0.0)
    qsp = torch.clamp((1.0 - fs) * qs, min=0.0)
    qgp = torch.clamp((1.0 - fg) * qg, min=0.0)
    qms = fs * (qr + qs)
    qmg = fg * (qr + qg)

    def _z_term(coeff: float, exp: float, q: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(q)
        mask = q > qeps
        out[mask] = coeff * torch.pow(ro[mask] * q[mask] * 1.0e3, exp)
        return out

    zr = _z_term(2.53e4, 1.84, qrp)
    zs = _z_term(3.48e3, 1.66, qsp)
    zg = _z_term(5.54e3, 1.70, qgp)

    fws = qr / (qr + qs + qeps)
    fwg = qr / (qr + qg + qeps)

    zms = torch.zeros_like(qms)
    mask_ms = qms > qeps
    zms[mask_ms] = (
        (0.00491 + 5.75 * fws[mask_ms] - 5.588 * fws[mask_ms] ** 2)
        * 1.0e5
        * torch.pow(
            ro[mask_ms] * qms[mask_ms] * 1.0e3,
            1.67 - 0.202 * fws[mask_ms] + 0.398 * fws[mask_ms] ** 2,
        )
    )

    zmg = torch.zeros_like(qmg)
    mask_mg = qmg > qeps
    zmg[mask_mg] = (
        (0.0358 + 5.27 * fwg[mask_mg] - 9.51 * fwg[mask_mg] ** 2 + 4.68 * fwg[mask_mg] ** 3)
        * 1.0e5
        * torch.pow(
            ro[mask_mg] * qmg[mask_mg] * 1.0e3,
            1.70 + 0.02 * fwg[mask_mg] + 0.287 * fwg[mask_mg] ** 2 - 0.186 * fwg[mask_mg] ** 3,
        )
    )

    if not use_melt:
        zms[:] = 0.0
        zmg[:] = 0.0

    return zr + zs + zg + zms + zmg