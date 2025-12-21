from __future__ import annotations

import numpy as np
import torch
import pytest
from xtensor import DataTensor, Dataset

from pyletkf.io.das_letkf_dumps import (
    load_das_letkf_core_before,
    load_das_postproc_after,
)
from pyletkf.letkf_core import letkf_core
from pyletkf.post_letkf import apply_analysis_update
from pyletkf.params import MEMBERS
from pyletkf_tests.helpers import (
    cell_index,
    _bool_from_meta,
    prepare_postproc_tensors,
)

CALL_IDS = [1, 12, 23]


class TensorWrapper:
    def __init__(self, tensor: torch.Tensor):
        self.values = tensor


def _build_core_dataset(hdxf, dep, rloc, rdiag, mask, parm_infl, rdiag_wloc, infl_update):
    obs_count = hdxf.shape[0]
    members = len(MEMBERS)
    batch_coord = torch.arange(1, dtype=torch.int64)
    obs_coord = torch.arange(obs_count, dtype=torch.int64)
    member_coord = torch.arange(members, dtype=torch.int64)
    coords = {"batch": batch_coord, "obs": obs_coord, "member": member_coord}
    data_vars = {
        "hdxf": DataTensor(
            hdxf.unsqueeze(0),
            coords,
            ("batch", "obs", "member"),
        ),
        "dep": DataTensor(
            dep.unsqueeze(0),
            {"batch": batch_coord, "obs": obs_coord},
            ("batch", "obs"),
        ),
        "rloc": DataTensor(
            rloc.unsqueeze(0),
            {"batch": batch_coord, "obs": obs_coord},
            ("batch", "obs"),
        ),
        "rdiag": DataTensor(
            rdiag.unsqueeze(0),
            {"batch": batch_coord, "obs": obs_coord},
            ("batch", "obs"),
        ),
        "obs_mask": DataTensor(
            mask.unsqueeze(0),
            {"batch": batch_coord, "obs": obs_coord},
            ("batch", "obs"),
        ),
        "parm_infl": DataTensor(
            torch.tensor([parm_infl], dtype=torch.float64),
            {"batch": batch_coord},
            ("batch",),
        ),
        "rdiag_wloc": DataTensor(
            torch.tensor([rdiag_wloc], dtype=torch.bool),
            {"batch": batch_coord},
            ("batch",),
        ),
        "infl_update": DataTensor(
            torch.tensor([infl_update], dtype=torch.bool),
            {"batch": batch_coord},
            ("batch",),
        ),
    }
    return Dataset(data_vars, coords={"batch": batch_coord, "obs": obs_coord, "member": member_coord})


@pytest.mark.parametrize("call_id", CALL_IDS)
def test_full_pipeline_matches_fortran(pre_letkf_dataset, state_dataset, dump_root, target_pe, call_id):
    core_before = load_das_letkf_core_before(
        dump_root, call_id=call_id, pe_tag=target_pe, member="mem0001"
    )
    meta = core_before["meta"]
    nlev = state_dataset["state"].sizes["z"]
    cell_idx = cell_index(int(meta["ij"]), int(meta["ilev"]), nlev)

    hdxf = pre_letkf_dataset["hdxf"].data[cell_idx]
    dep = pre_letkf_dataset["dep"].data[cell_idx]
    rloc = pre_letkf_dataset["rloc"].data[cell_idx]
    rdiag = pre_letkf_dataset["rdiag"].data[cell_idx]
    mask = pre_letkf_dataset["obs_mask"].data[cell_idx]

    nobs = core_before["data"]["hdxf"].shape[0]
    indices = torch.nonzero(mask.to(torch.bool), as_tuple=False).squeeze(1)[:nobs]

    hdxf_sel = hdxf[indices].to(torch.float64)
    dep_sel = dep[indices].to(torch.float64)
    rloc_sel = rloc[indices].to(torch.float64)
    rdiag_sel = rdiag[indices].to(torch.float64)
    mask_sel = torch.ones_like(dep_sel, dtype=torch.bool)

    batch_ds = _build_core_dataset(
        hdxf_sel,
        dep_sel,
        rloc_sel,
        rdiag_sel,
        mask_sel,
        float(meta["parm_infl"]),
        _bool_from_meta(meta["rdiag_wloc"]),
        _bool_from_meta(meta["infl_update"]),
    )

    core_out = letkf_core(batch_ds)
    trans = core_out["trans"].data[0]
    transm = core_out["transm"].data[0]
    parm_infl = core_out["parm_infl"].data[0]

    post_data, params, post_after = prepare_postproc_tensors(
        call_id, dump_root, target_pe
    )

    tensor_data = {
        "trans": TensorWrapper(trans.unsqueeze(0)),
        "transm": TensorWrapper(transm.unsqueeze(0)),
        "gues_members": TensorWrapper(post_data.gues_members),
        "gues_mean": TensorWrapper(post_data.gues_mean),
        "parm_infl": TensorWrapper(parm_infl.unsqueeze(0)),
    }
    tensor_params = {
        "beta": TensorWrapper(params.beta),
        "relax_alpha": TensorWrapper(params.relax_alpha),
        "relax_alpha_spread": TensorWrapper(params.relax_alpha_spread),
        "relax_to_inflated": TensorWrapper(params.relax_to_inflated),
        "relax_spread_out": TensorWrapper(params.relax_spread_out),
        "det_run": TensorWrapper(params.det_run),
        "nvar": TensorWrapper(params.nvar.to(torch.float64)),
    }

    outputs = apply_analysis_update(tensor_data, tensor_params)
    np.testing.assert_allclose(
        outputs["anal_members"].cpu().numpy()[0],
        post_after["data"]["anal_members"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
