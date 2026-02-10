import torch

def scale_to_letkf(state, rdry, cvap, rvap, cvdry, pre00):
    """
    """
    rho, rhot = state[:2]                   # DENS, RHO
    moments = state[2:5] / rho              # u, v, w
    moist = torch.nan_to_num(state[5:], 0)  # QV, QC, QR, QI, QS, QG

    qdry = 1.0 - moist.sum(dim=0)
    rtot = rdry * qdry + rvap * moist[0]
    cv_tot = cvdry * qdry + (moist * cvap).sum(dim=0)
    base = (rhot * rtot) / pre00

    valid = (base > 0.0) & (rho > 0.0) & (cv_tot > 0.0) & (rtot > 0.0)

    gamma = torch.where(valid, (cv_tot + rtot) / cv_tot, torch.nan)
    pressure = torch.where(valid, pre00 * torch.pow(base, gamma), torch.nan)
    temperature = torch.where(valid, pressure / (rho * rtot), torch.nan)

    return torch.cat([moments, temperature[None], pressure[None], moist])

def letkf_to_scale(state, rdry, cvap, rvap, cvdry, pre00):
    """
    """
    moments = state[:3]                     # u, v, w
    temperature = state[3]                  # T
    pressure = state[4]                     # p
    moist = torch.nan_to_num(state[5:], 0)  # QV, QC, QR, QI, QS, QG

    qdry = 1.0 - moist.sum(dim=0)
    rtot = rdry * qdry + rvap * moist[0]
    cv_tot = cvdry * qdry + (moist * cvap).sum(dim=0)

    valid = (pressure > 0.0) & (temperature > 0.0) & (cv_tot > 0.0) & (rtot > 0.0)

    gamma = torch.where(valid, (cv_tot + rtot) / cv_tot, torch.nan)
    base = torch.where(valid, pressure / pre00, torch.nan)
    base = torch.where(valid, torch.pow(base, torch.reciprocal(gamma)), torch.nan)

    rhot = torch.where(valid, base * pre00 / rtot, torch.nan)
    rho = torch.where(valid, pressure / (rtot * temperature), torch.nan)

    mom = moments * rho
    return torch.cat([rho[None], rhot[None], mom, moist])