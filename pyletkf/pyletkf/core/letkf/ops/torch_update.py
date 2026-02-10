import torch

def relax_rtpp(Wa, alpha):
    """
        Wa: [batch, state_cell, ensemble, ensemble]
        xb: [batch, state_cell, ensemble]
        alpha_spread: int
    """
    if alpha==0: return Wa
    i = torch.eye(Wa.shape[-1], 
                  device=Wa.device, 
                  dtype=Wa.dtype).view(1, 1, K, K)
    return (1.0 - alpha) * Wa + alpha_pert * i
    
def relax_rtps(Wa, xb, alpha, eps=1e-12):
    """
        Wa: [batch, state_cell, ensemble, ensemble]
        xb: [batch, state_cell, ensemble]
        alpha_spread: int
    """
    if alpha == 0: return Wa
    xa = torch.bmm(
            xb.view(B*S, 1, K),
            Wa.view(B*S, K, K)
        ).view(B, S, K)
    sigma_b = xb.std(dim=-1, unbiased=True) # [B,S]
    sigma_a = xa.std(dim=-1, unbiased=True) # [B,S]
    gamma = 1.0 + alpha * (sigma_b - sigma_a) / (sigma_a + eps)
    return Wa * gamma.unsqueeze(-1).unsqueeze(-1)

def letkf_update(xb, wa, Wa, alpha_pert, alpha_spread, beta):
    """
    """
    Wa = relax_rtpp(Wa, alpha_pert)
    Wa = relax_rtps(Wa, xb, alpha_spread)
    
    w = (Wa + wa.unsqueeze(-1)) 
    w = w * beta + torch.eye(w.shape[-1], 
                             device=w.device)[None] \
                   * (1.0 - beta)
    xa = torch.bmm(xb, w)
    return xa