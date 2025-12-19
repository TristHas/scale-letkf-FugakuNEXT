from __future__ import annotations
import math
from dataclasses import dataclass

GRAV = 9.81  # m/s^2
PI = math.pi

@dataclass
class ReflectivityContext:
    method: int = 3
    use_terminal_velocity: bool = True
    use_t08_rs2014: bool = False
    radar_adaptive_y18: bool = False
    min_radar_ref: float = 10.0  # mm^6/m^3
    min_radar_ref_dbz: float = 10.0
    low_ref_shift: float = -5.0

def _gamma(value: float) -> float:
    return math.gamma(value)

def calc_ref_vr(
    qv: float,
    qc: float,
    qr: float,
    qi: float,
    qs: float,
    qg: float,
    u: float,
    v: float,
    w: float,
    temp: float,
    press: float,
    azimuth_deg: float,
    elev_deg: float,
    ctx: ReflectivityContext | None = None,
) -> tuple[float, float]:
    """Reproduce the SCALE ``calc_ref_vr`` (METHOD_REF_CALC=3) operator."""

    ctx = ctx or ReflectivityContext()
    if ctx.method != 3:
        raise NotImplementedError("Only METHOD_REF_CALC=3 is supported")

    ro = max(press / (287.04 * temp), 1.0e-12)
    qeps = 1.0e-20
    maxf = 0.5

    def _safe_ratio(num: float, den: float) -> float:
        return num / den if den > qeps else 0.0

    if qr > qeps and qg > qeps:
        fg = maxf * min(_safe_ratio(qr, qg), _safe_ratio(qg, qr)) ** (1.0 / 3.0)
    else:
        fg = 0.0
    if qr > qeps and qs > qeps:
        fs = maxf * min(_safe_ratio(qr, qs), _safe_ratio(qs, qr)) ** (1.0 / 3.0)
    else:
        fs = 0.0

    qrp = max(0.0, (1.0 - fs - fg) * qr)
    qsp = max(0.0, (1.0 - fs) * qs)
    qgp = max(0.0, (1.0 - fg) * qg)
    qms = fs * (qr + qs)
    qmg = fg * (qr + qg)

    def _z_power(coeff: float, exponent: float, mix_ratio: float) -> float:
        if mix_ratio <= qeps:
            return 0.0
        return coeff * (ro * mix_ratio * 1.0e3) ** exponent

    zr = _z_power(2.53e4, 1.84, qrp)
    zs = _z_power(3.48e3, 1.66, qsp)
    zg = _z_power(5.54e3, 1.70, qgp)

    fws = qr / (qr + qs + qeps)
    fwg = qr / (qr + qg + qeps)

    zms = 0.0
    if qms > qeps:
        zms = (
            (0.00491 + 5.75 * fws - 5.588 * fws**2)
            * 1.0e5
            * (ro * qms * 1.0e3) ** (1.67 - 0.202 * fws + 0.398 * fws**2)
        )
    zmg = 0.0
    if qmg > qeps:
        zmg = (
            (0.0358 + 5.27 * fwg - 9.51 * fwg**2 + 4.68 * fwg**3)
            * 1.0e5
            * (ro * qmg * 1.0e3) ** (1.70 + 0.02 * fwg + 0.287 * fwg**2 - 0.186 * fwg**3)
        )

    radar_ref = max(zr + zs + zg + zms + zmg, 0.0)

    wt = 0.0
    if radar_ref > 0.0:
        nor = 8.0e-2
        nos = 3.0e-2
        nog = 4.0e-2
        ror = 1.0
        ros = 0.1
        rog = 0.4
        roo = 1.28e-3
        cd = 0.6

        ro_cgs = max(ro * 1.0e-3, 1.0e-12)
        rofactor = math.sqrt(roo / ro_cgs)

        dr = 0.5
        ds = 0.25
        dg = 0.5
        cr = 130.0
        cs = 4.84

        def _vel_term(mix: float, no: float, rho_s: float, coeff: float, exponent: float) -> float:
            if mix <= qeps:
                return 0.0
            lam = (PI * rho_s * no / (ro_cgs * mix)) ** 0.25
            return coeff * _gamma(4.0 + exponent) / (6.0 * (lam * 1.0e2) ** exponent) * rofactor

        wr = _vel_term(qr, ror, nor, cr, dr)
        ws = _vel_term(qs, ros, nos, cs, ds)

        if qg > qeps:
            lg = (PI * rog * nog / (ro_cgs * qg)) ** 0.25
            tmp = _gamma(4.0 + dg)
            wg = (
                tmp
                * math.sqrt((4.0 * GRAV * rog) / (3.0 * cd * roo))
                / (6.0 * (lg * 1.0e2) ** dg)
                * rofactor
            )
        else:
            wg = 0.0

        denominator = zr + zs + zg + zms + zmg
        if denominator > 0.0:
            wt = (wr * zr + ws * (zs + zms) + wg * (zg + zmg)) / denominator

    az = math.radians(azimuth_deg)
    el = math.radians(elev_deg)
    vr = (
        u * math.cos(el) * math.sin(az)
        + v * math.cos(el) * math.cos(az)
    )
    if ctx.use_terminal_velocity:
        vr += (w - wt) * math.sin(el)
    else:
        vr += w * math.sin(el)

    return radar_ref, vr


__all__ = ["ReflectivityContext", "calc_ref_vr"]
