import argparse
import math
import sys
from pathlib import Path

import torch

CURRENT_DIR = Path(__file__).resolve().parent
SC23_DIR = CURRENT_DIR.parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
PYLETKF_ROOT = PROJECT_ROOT / "pyletkf"

for path in (SC23_DIR, CURRENT_DIR, PROJECT_ROOT, PYLETKF_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from sc23 import init_sc23  # noqa: E402
from test_io.letkf_dumps import load_obs_interp_dump  # noqa: E402

VAR_ORDER = ["U", "V", "W", "T", "P", "QV", "QC", "QR", "QI", "QS", "QG"]

def itpl_3d_fortran(var, ri, rj, rk):
    """
    Reproduce SCALE's itpl_3d routine assuming var is ordered as (z, x, y).
    """
    nz, nx, ny = var.shape
    i = math.ceil(ri)
    j = math.ceil(rj)
    k = math.ceil(rk)
    ai = ri - float(i - 1)
    aj = rj - float(j - 1)
    ak = rk - float(k - 1)

    def clamp(idx, upper):
        return max(1, min(idx, upper))

    def sample(kk, ii, jj):
        kk = clamp(kk, nz)
        ii = clamp(ii, nx)
        jj = clamp(jj, ny)
        return var[kk - 1, ii - 1, jj - 1]

    if nx == 1:
        # reduced expression used in Fortran for 2D ideal cases
        return (
            sample(k - 1, i, j - 1) * (1.0 - aj) * (1.0 - ak)
            + sample(k - 1, i, j) * aj * (1.0 - ak)
            + sample(k, i, j - 1) * (1.0 - aj) * ak
            + sample(k, i, j) * aj * ak
        )

    return (
        sample(k - 1, i - 1, j - 1) * (1.0 - ai) * (1.0 - aj) * (1.0 - ak)
        + sample(k - 1, i, j - 1) * ai * (1.0 - aj) * (1.0 - ak)
        + sample(k - 1, i - 1, j) * (1.0 - ai) * aj * (1.0 - ak)
        + sample(k - 1, i, j) * ai * aj * (1.0 - ak)
        + sample(k, i - 1, j - 1) * (1.0 - ai) * (1.0 - aj) * ak
        + sample(k, i, j - 1) * ai * (1.0 - aj) * ak
        + sample(k, i - 1, j) * (1.0 - ai) * aj * ak
        + sample(k, i, j) * ai * aj * ak
    )

def load_state(rank):
    grid, pipeline = init_sc23()
    tile = grid[rank]
    device = torch.device("cpu")
    state = tile.read_state().to(device)
    state = tile.share_halos(state)
    return pipeline.state_conv.model_to_da(state)

def select_entries(rank, obs_idx):
    entries = load_obs_interp_dump()
    subset = [
        entry
        for entry in entries
        if entry["label"] == "radar_state"
        and entry["stage"] == "obsop"
        and entry["rank_local"] == rank
        and entry["obs_idx"] == obs_idx
    ]
    if not subset:
        raise ValueError(f"No dump rows for rank={rank}, obs_idx={obs_idx}")
    subset.sort(key=lambda e: e["rank_global"])
    return subset

def main():
    parser = argparse.ArgumentParser(description="Reproduce SCALE interpolation for one observation.")
    parser.add_argument("--rank", type=int, default=5, help="Tile rank to load.")
    parser.add_argument("--obs", type=int, default=36523, help="Observation index to inspect.")
    args = parser.parse_args()

    state = load_state(args.rank)
    ens_labels = list(state["ens"].values)
    var_lookup = {str(name): idx for idx, name in enumerate(state["variable"].values)}
    tensor = state["state"].values.to(torch.float64).contiguous()
    entries = select_entries(args.rank, args.obs)

    grid, _ = init_sc23()
    n_tiles = grid.n_tiles_x * grid.n_tiles_y

    print(f"Observation {args.obs} on rank {args.rank}")
    print(f"ri={entries[0]['ri']:.6f}, rj={entries[0]['rj']:.6f}, rk={entries[0]['rk']:.6f}")
    print()

    for entry in entries:
        ens_idx = entry["rank_global"] // n_tiles
        ens_label = ens_labels[ens_idx] if ens_idx < len(ens_labels) else f"ens{ens_idx}"
        print(f"rank_global={entry['rank_global']} -> ensemble {ens_label}")
        for var_name, ft_value in zip(VAR_ORDER, entry["values"]):
            var_idx = var_lookup[var_name]
            field = (
                tensor[ens_idx, var_idx]
                .permute(2, 1, 0)
                .contiguous()
                .numpy()
            )
            py_value = itpl_3d_fortran(field, entry["ri"], entry["rj"], entry["rk"])
            diff = py_value - ft_value
            print(f"  {var_name:<2} fortran={ft_value: .6e}  python={py_value: .6e}  diff={diff: .6e}")
        print()


if __name__ == "__main__":
    main()
