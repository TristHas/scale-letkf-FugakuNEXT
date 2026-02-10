import pytest
import torch

torch_ops = pytest.importorskip("pyletkf.model_state.ops.torch")
triton_ops = pytest.importorskip("pyletkf.model_state.ops.triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for Triton ops tests", allow_module_level=True)

DEVICE = torch.device("cuda")
DTYPES = (torch.float16, torch.float32, torch.float64)
ENSEMBLE_SIZES = (2, 8)
GRID_SIZES = (10, 20)

RD_RY = 287.04
RV_AP = 461.50
CP_DRY = 1004.64
CP_VAP = 1846.00
CV_DRY = CP_DRY - RD_RY
CV_VAP = CP_VAP - RV_AP
PRE00 = 100000.0

SCALE_VAR_COUNT = 11
NUM_MOIST = 6

TOLERANCES = {
    torch.float16: {"rtol": 5e-2, "atol": 5e-2},
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-6, "atol": 1e-5},
}


def _make_scale_state(dtype: torch.dtype, ensemble: int, grid: int) -> torch.Tensor:
    shape = (SCALE_VAR_COUNT, ensemble, grid, grid)
    state = torch.zeros(shape, dtype=dtype, device=DEVICE)

    state[0] = torch.rand_like(state[0]) + 1.0  # density
    state[1] = torch.rand_like(state[1]) + 1.0  # rhot
    state[2:5] = torch.randn_like(state[2:5])   # momenta
    moist = torch.rand((NUM_MOIST, ensemble, grid, grid), dtype=dtype, device=DEVICE) * 1e-3
    state[5:] = moist
    return state


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ensemble", ENSEMBLE_SIZES)
@pytest.mark.parametrize("grid", GRID_SIZES)
def test_scale_to_letkf_matches_torch(dtype, ensemble, grid):
    state = _make_scale_state(dtype, ensemble, grid)

    expected = torch_ops.scale_to_letkf(state.clone(), RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)
    result = triton_ops.scale_to_letkf(state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(result, expected, equal_nan=True, **tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ensemble", ENSEMBLE_SIZES)
@pytest.mark.parametrize("grid", GRID_SIZES)
def test_letkf_to_scale_matches_torch(dtype, ensemble, grid):
    scale_state = _make_scale_state(dtype, ensemble, grid)
    letkf_state = torch_ops.scale_to_letkf(scale_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)

    expected = torch_ops.letkf_to_scale(letkf_state.clone(), RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)
    result = triton_ops.letkf_to_scale(letkf_state, RD_RY, CV_VAP, RV_AP, CV_DRY, PRE00)

    tol = TOLERANCES[dtype]
    torch.testing.assert_close(result, expected, equal_nan=True, **tol)
