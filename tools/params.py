from typing import Dict

# Run-time invariants taken from test/SC23/conf/letkf_20210730060030.conf
MEMBER = 2          # ensemble size requested in PARAM_ENSEMBLE
DET_RUN = False     # deterministic member disabled
NENS = MEMBER + (2 if DET_RUN else 1)
# Mirrors common_mpi_scale::set_mem_node_proc logic for rank bookkeeping.
TOTAL_RANKS = 60    # mpiexec -n value in test/SC23/exec_timed.sh
PROCS_PER_MEMBER = 20  # PRC_DOMAINS for the single domain
M_MEAN_RANK_E = MEMBER          # ensemble-mean communicator rank
MPI_COMM_A_COLOR = 0            # only one physical domain

LETKF_CONSTANTS = {
  "MEMBER": MEMBER,
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

GRID_CONSTANTS = {
  "DX": 100.0,
  "DY": 100.0,
  "nlon": 320,
  "nlat": 256,
  "nlev": 45,
  "nv3d": 11,
  "nv2d": 0,
}

OBS_ID_CONSTANTS = {
  "id_ps_obs": 14593,
  "id_rain_obs": 19999,
  "id_radar_ref_obs": 4001,
}

DA_CONSTANTS = {
  "IHALO": 2,
  "JHALO": 2,
  "BOUNDARY_BUFFER_WIDTH": 1.0e4,
  "RADAR_ZMAX": 11000.0,
  "RADAR_ONLY": False,
  "VERT_LOCAL_RADAR": 2000.0,
}

def rank_properties(global_rank: int) -> Dict[str, int]:
    """Return per-rank identifiers used inside das_letkf."""
    if not 0 <= global_rank < TOTAL_RANKS:
        raise ValueError("rank must be between 0 and 59")
    member_slot = global_rank // PROCS_PER_MEMBER
    pe_slot = global_rank % PROCS_PER_MEMBER
    member_id = member_slot + 1  # 1..3 (member 1, member 2, mean)
    return {
        "member_id": member_id,
        "pe_id": pe_slot,          # PRC_DOMAINS index
        "myrank_e": member_slot,   # ensemble communicator rank
        "myrank_a": member_slot * PROCS_PER_MEMBER + pe_slot,
        "MPI_COMM_e": pe_slot,     # communicator color shared by same pe_id
        "MPI_COMM_a": MPI_COMM_A_COLOR,
        "mmean_rank_e": M_MEAN_RANK_E,
    }
