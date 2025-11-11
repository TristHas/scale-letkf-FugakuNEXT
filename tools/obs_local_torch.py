"""Torch-friendly wrapper for ``obs_local``.

Current implementation delegates to the validated NumPy version so that the
Torch pipeline can reuse the same interface while we continue porting and
vectorising the localisation stage.
"""

from __future__ import annotations

import numpy as np

from .obs_local import (
    ObsLocalIdentifier,
    ObsLocalInputs,
    load_obs_local_from_global,
    obs_local,
)


def obs_local_torch(
    inputs: ObsLocalInputs,
    search_q0: np.ndarray | None = None,
    *,
    device=None,
):
    search_buf = None
    if search_q0 is not None:
        search_buf = np.array(search_q0, dtype=np.int64, copy=True)
    output = obs_local(inputs, search_buf)
    if search_buf is not None and search_q0 is not None:
        search_q0[:] = search_buf
    return output


def load_obs_local_torch(rank_identifier: ObsLocalIdentifier | dict) -> ObsLocalInputs:
    return load_obs_local_from_global(rank_identifier)


__all__ = ["obs_local_torch", "load_obs_local_torch"]
