from __future__ import annotations

import numpy as np
import torch

from pyletkf.post_letkf import apply_analysis_update
from pyletkf_tests.helpers import prepare_postproc_tensors, _bool_from_meta


class TensorWrapper:
    def __init__(self, tensor: torch.Tensor):
        self.values = tensor


def _build_simple_container(tensors: dict[str, torch.Tensor]):
    return {key: TensorWrapper(value) for key, value in tensors.items()}


def test_postproc_matches_fortran(dump_root, target_pe):
    call_id = 1
    data_tensors, params, after = prepare_postproc_tensors(call_id, dump_root, target_pe)
    data = _build_simple_container(
        dict(
            trans=data_tensors.trans,
            transm=data_tensors.transm,
            gues_members=data_tensors.gues_members,
            gues_mean=data_tensors.gues_mean,
            parm_infl=data_tensors.parm_infl,
        )
    )
    param = _build_simple_container(
        dict(
            beta=params.beta,
            relax_alpha=params.relax_alpha,
            relax_alpha_spread=params.relax_alpha_spread,
            relax_to_inflated=params.relax_to_inflated,
            relax_spread_out=params.relax_spread_out,
            det_run=params.det_run,
            nvar=params.nvar,
        )
    )
    outputs = apply_analysis_update(data, param)
    np.testing.assert_allclose(
        outputs["transrlx"].cpu().numpy()[0],
        after["data"]["transrlx"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        outputs["anal_members"].cpu().numpy()[0],
        after["data"]["anal_members"],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert bool(outputs["q_limited"].cpu().numpy()[0]) == _bool_from_meta(
        after["meta"]["q_limited"]
    )
    np.testing.assert_allclose(
        outputs["q_mean"].cpu().numpy()[0],
        float(after["meta"]["q_mean"]),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        outputs["workda_value"].cpu().numpy()[0],
        float(after["meta"]["workda_value"]),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert bool(outputs["workda_present"].cpu().numpy()[0]) == _bool_from_meta(
        after["meta"]["workda_present"]
    )
