import torch

def convert_raw_to_dbz(raw, min_radar_ref_dbz=10.0, low_res_shift=-5.):
    """
    """
    valid = (raw >= 0.0) & (raw < 1.0e10)
    dbz = torch.full_like(raw, float('nan'))
    # Segment clear-sky vs rest
    low_mask  = valid & (raw < 10.0 ** (min_radar_ref_dbz / 10.0))
    high_mask = valid & (~low_mask)
    # Clear-sky: set fixed DBZ
    dbz = torch.where(low_mask,  min_radar_ref_dbz + low_res_shift, dbz)
    # Others: convert linear reflectivity → dBZ
    dbz = torch.where(high_mask, 10.0 * torch.log10(raw), dbz)
    return dbz

def _mix_ratio_ratio(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """
    """
    ratio = torch.zeros_like(a)
    valid = (a > eps) & (b > eps)
    safe = torch.minimum(a[valid] / b[valid], b[valid] / a[valid])
    ratio[valid] = safe
    return ratio

def _safe_fraction(numer: torch.Tensor, denom: torch.Tensor, eps: float) -> torch.Tensor:
    """
    """
    frac = torch.zeros_like(numer)
    valid = (denom.abs() > eps)
    frac[valid] = numer[valid] / denom[valid]
    return frac

def linear_reflectivity_method3(
        qr: torch.Tensor,
        qs: torch.Tensor,
        qg: torch.Tensor,
        temp: torch.Tensor,
        press: torch.Tensor,
        *,
        use_melt: bool,
        use_t08_rs2014: bool,
        qeps: float = 1.0e-20,
        return_terms: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    """
    ro = press / (287.04 * temp)
    maxf = 0.5

    if use_melt:
        fg = maxf * torch.pow(_mix_ratio_ratio(qr, qg, qeps), 1.0 / 3.0)
        fs = maxf * torch.pow(_mix_ratio_ratio(qr, qs, qeps), 1.0 / 3.0)
        fwg = _safe_fraction(qr, qr + qg, qeps)
        fws = _safe_fraction(qr, qr + qs, qeps)
    else:
        fg = torch.zeros_like(qr)
        fs = torch.zeros_like(qr)
        fwg = torch.zeros_like(qr)
        fws = torch.zeros_like(qr)

    qrp = torch.clamp((1.0 - fs - fg) * qr, min=0.0)
    qsp = torch.clamp((1.0 - fs) * qs, min=0.0)
    qgp = torch.clamp((1.0 - fg) * qg, min=0.0)
    qms = torch.clamp(fs * (qr + qs), min=0.0)
    qmg = torch.clamp(fg * (qr + qg), min=0.0)

    def _z_term(coeff: float, exp: float, q: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(q)
        mask = q > qeps
        out[mask] = coeff * torch.pow(ro[mask] * q[mask] * 1.0e3, exp)
        return out

    zr = _z_term(2.53e4, 1.84, qrp)
    zs = _z_term(3.48e3, 1.66, qsp)
    zg = _z_term(5.54e3, 1.70, qgp)

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
            1.70 + 0.020 * fwg[mask_mg] + 0.287 * fwg[mask_mg] ** 2 - 0.186 * fwg[mask_mg] ** 3,
        )
    )

    if not use_melt:
        zms[:] = 0.0
        zmg[:] = 0.0

    radar_lin = zr + zs + zg + zms + zmg
    if not return_terms:
        return radar_lin
    return radar_lin, {
        "density": ro,
        "zr": zr,
        "zs": zs,
        "zg": zg,
        "zms": zms,
        "zmg": zmg,
        "qr": qr,
        "qs": qs,
        "qg": qg,
    }
