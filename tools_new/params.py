NENS = 2
MEMBERS = ("0001", "0002")

LETKF_CONSTANTS = {
  "MEMBER": len(MEMBERS),
  "NENS": NENS,
  "INFL_MUL_ADAPTIVE": False,
  "INFL_ADD": 0.0,
  "INFL_ADD_Q_RATIO": False,
  "RELAX_ALPHA": 0.95,
  "RELAX_ALPHA_SPREAD": 0.0,
  "RELAX_SPREAD_OUT": False,
  "dist_zero_fac": 3.651483717,
  "Q_UPDATE_TOP": 30000.0,
  "Q_SPRD_MAX": 0.5,
  # DET_RUN is false, so no deterministic member is formed.
  "mmdet": None,
  "mmdetin": None,
}
