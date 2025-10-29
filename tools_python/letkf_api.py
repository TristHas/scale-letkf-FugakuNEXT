from __future__ import annotations

import ctypes
from ctypes import c_int, c_double, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

LIB_RELATIVE = Path(__file__).resolve().parents[1] / "scale/scale-letkf/scale/letkf/libletkf_api.so"


@dataclass
class CommonNMLInfo:
    member: int
    det_run: bool
    obsda_in: bool
    log_level: int
    gues_sprd_out: bool
    anal_sprd_out: bool
    departure_stat: bool


@dataclass
class RankInfo:
    rank: int
    use: bool


class LetkfAPI:
    """Thin ctypes bindings around the LETKF C ABI."""

    STATE_3D = 1
    STATE_2D = 2

    OBS_INT_FIELDS: Dict[str, int] = {
        "elm": 1,
        "typ": 7,
        "rank": 13,
    }
    OBS_REAL_FIELDS: Dict[str, int] = {
        "lon": 2,
        "lat": 3,
        "lev": 4,
        "dat": 5,
        "err": 6,
        "dif": 8,
        "meta": 9,
        "ri": 10,
        "rj": 11,
    }

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        self.lib_path = Path(lib_path) if lib_path else LIB_RELATIVE
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._define_prototypes()
        self._initialized = False

    # ------------------------------------------------------------------
    # public lifecycle -------------------------------------------------
    def initialize(self) -> None:
        ierr = c_int(0)
        self.lib.letkf_initialize_mpi_scale(ctypes.byref(ierr))
        self._check(ierr.value, "letkf_initialize_mpi_scale")
        self._initialized = True

    def finalize(self) -> None:
        if not self._initialized:
            return
        ierr = c_int(0)
        self.lib.letkf_finalize_mpi_scale(ctypes.byref(ierr))
        self._check(ierr.value, "letkf_finalize_mpi_scale")
        self._initialized = False

    # ------------------------------------------------------------------
    # configuration helpers -------------------------------------------
    def get_common_nml_info(self) -> CommonNMLInfo:
        member = c_int()
        det_run = c_int()
        obsda_in = c_int()
        log_level = c_int()
        gues_sprd = c_int()
        anal_sprd = c_int()
        departure_stat = c_int()
        ierr = c_int()
        self.lib.letkf_get_common_nml_info(
            ctypes.byref(member),
            ctypes.byref(det_run),
            ctypes.byref(obsda_in),
            ctypes.byref(log_level),
            ctypes.byref(gues_sprd),
            ctypes.byref(anal_sprd),
            ctypes.byref(departure_stat),
            ctypes.byref(ierr),
        )
        self._check(ierr.value, "letkf_get_common_nml_info")
        return CommonNMLInfo(
            member=member.value,
            det_run=bool(det_run.value),
            obsda_in=bool(obsda_in.value),
            log_level=log_level.value,
            gues_sprd_out=bool(gues_sprd.value),
            anal_sprd_out=bool(anal_sprd.value),
            departure_stat=bool(departure_stat.value),
        )

    def get_mpi_rank_info(self) -> RankInfo:
        rank = c_int()
        use_flag = c_int()
        ierr = c_int()
        self.lib.letkf_get_mpi_rank_info(
            ctypes.byref(rank), ctypes.byref(use_flag), ctypes.byref(ierr)
        )
        self._check(ierr.value, "letkf_get_mpi_rank_info")
        return RankInfo(rank=rank.value, use=bool(use_flag.value))

    # ------------------------------------------------------------------
    # wrappers for individual steps -----------------------------------
    def set_config_path(self, path: str) -> None:
        encoded = path.encode("utf-8")
        self._invoke(
            self.lib.letkf_set_config_path,
            ctypes.c_char_p(encoded),
            name="letkf_set_config_path",
        )

    def set_mem_node_proc(self, mem: int) -> None:
        self._invoke(self.lib.letkf_set_mem_node_proc, c_int(mem), name="letkf_set_mem_node_proc")

    def set_scalelib(self, name: str) -> None:
        encoded = name.encode("utf-8")
        self._invoke(
            self.lib.letkf_set_scalelib,
            ctypes.c_char_p(encoded),
            name="letkf_set_scalelib",
        )

    def unset_scalelib(self) -> None:
        self._invoke(self.lib.letkf_unset_scalelib, name="letkf_unset_scalelib")

    def set_common_scale(self) -> None:
        self._invoke(self.lib.letkf_set_common_scale, name="letkf_set_common_scale")

    def set_common_mpi_scale(self) -> None:
        self._invoke(self.lib.letkf_set_common_mpi_scale, name="letkf_set_common_mpi_scale")

    def unset_common_mpi_scale(self) -> None:
        self._invoke(self.lib.letkf_unset_common_mpi_scale, name="letkf_unset_common_mpi_scale")

    def set_common_obs_scale(self) -> None:
        self._invoke(self.lib.letkf_set_common_obs_scale, name="letkf_set_common_obs_scale")

    def mtx_setup(self, member: int) -> None:
        self._invoke(self.lib.letkf_mtx_setup, c_int(member), name="letkf_mtx_setup")

    def read_obs_all_mpi(self) -> None:
        self._invoke(self.lib.letkf_read_obs_all_mpi, name="letkf_read_obs_all_mpi")

    def get_nobs_da_mpi(self) -> int:
        nobs = c_int()
        ierr = c_int()
        self.lib.letkf_get_nobs_da_mpi(ctypes.byref(nobs), ctypes.byref(ierr))
        self._check(ierr.value, "letkf_get_nobs_da_mpi")
        return nobs.value

    def set_nobs_extern(self, value: int) -> None:
        self._invoke(self.lib.letkf_set_nobs_extern, c_int(value), name="letkf_set_nobs_extern")

    def obsope_cal(self) -> None:
        self._invoke(self.lib.letkf_obsope_cal, name="letkf_obsope_cal")

    def set_letkf_obs(self) -> None:
        self._invoke(self.lib.letkf_set_letkf_obs, name="letkf_set_letkf_obs")

    def dump_letkf_obs_state(self) -> None:
        self._invoke(self.lib.letkf_dump_letkf_obs_state, name="letkf_dump_letkf_obs_state")

    def dump_letkf_gues_state(self) -> None:
        self._invoke(self.lib.letkf_dump_letkf_gues_state, name="letkf_dump_letkf_gues_state")

    def set_common_mpi_grid(self) -> None:
        self._invoke(self.lib.letkf_set_common_mpi_grid, name="letkf_set_common_mpi_grid")

    def allocate_state_arrays(self) -> None:
        self._invoke(self.lib.letkf_allocate_state_arrays, name="letkf_allocate_state_arrays")

    def read_ens_mpi(self) -> None:
        self._invoke(self.lib.letkf_read_ens_mpi, name="letkf_read_ens_mpi")

    def adjust_det_run(self) -> None:
        self._invoke(self.lib.letkf_adjust_det_run, name="letkf_adjust_det_run")

    def write_gues_sprd(self) -> None:
        self._invoke(self.lib.letkf_write_enssprd, name="letkf_write_enssprd")

    def write_anal_sprd(self) -> None:
        self._invoke(self.lib.letkf_write_anal_sprd, name="letkf_write_anal_sprd")

    def das_letkf(self) -> None:
        self._invoke(self.lib.letkf_das_letkf, name="letkf_das_letkf")

    def dump_letkf_analysis_state(self) -> None:
        self._invoke(self.lib.letkf_dump_letkf_analysis_state, name="letkf_dump_letkf_analysis_state")

    def ensmean_grd(self, member: int, nens: int, nij1: int) -> None:
        self._invoke(
            self.lib.letkf_ensmean_grd,
            c_int(member),
            c_int(nens),
            c_int(nij1),
            name="letkf_ensmean_grd",
        )

    def write_ens_mpi(self, monit_step: Optional[int] = None) -> None:
        step = 0 if monit_step is None else monit_step
        self._invoke(self.lib.letkf_write_ens_mpi, c_int(step), name="letkf_write_ens_mpi")

    def write_ensmean(self, calced: bool, monit_step: Optional[int] = None) -> None:
        calced_flag = 1 if calced else 0
        step = 0 if monit_step is None else monit_step
        self._invoke(
            self.lib.letkf_write_ensmean,
            c_int(calced_flag),
            c_int(step),
            name="letkf_write_ensmean",
        )

    def mpi_timer(self, name: str, level: int, barrier: Optional[int] = None) -> None:
        barrier_val = -1 if barrier is None else barrier
        self._invoke(
            self.lib.letkf_mpi_timer,
            ctypes.c_char_p(name.encode("utf-8")),
            c_int(level),
            c_int(barrier_val),
            name="letkf_mpi_timer",
        )

    # ------------------------------------------------------------------
    # observation helpers ----------------------------------------------
    def get_obs_in_num(self) -> int:
        count = c_int()
        self.lib.letkf_get_obs_in_num(ctypes.byref(count))
        return count.value

    def get_obs_nobs(self, iof: int) -> int:
        nobs = c_int()
        ierr = c_int()
        self.lib.letkf_get_obs_nobs(c_int(iof), ctypes.byref(nobs), ctypes.byref(ierr))
        self._check(ierr.value, "letkf_get_obs_nobs")
        return nobs.value

    def get_obs_component_length(self, iof: int, component_id: int) -> int:
        length = c_int()
        ierr = c_int()
        self.lib.letkf_get_obs_component_length(
            c_int(iof), c_int(component_id), ctypes.byref(length), ctypes.byref(ierr)
        )
        self._check(ierr.value, "letkf_get_obs_component_length")
        return length.value

    def copy_obs_int_component(self, iof: int, component_id: int) -> np.ndarray:
        length = self.get_obs_component_length(iof, component_id)
        arr = np.empty(length, dtype=np.int32)
        if length > 0:
            self._invoke(
                self.lib.letkf_copy_obs_int_component,
                c_int(iof),
                c_int(component_id),
                c_void_p(arr.ctypes.data),
                c_int(length),
                name="letkf_copy_obs_int_component",
            )
        return arr

    def copy_obs_real_component(self, iof: int, component_id: int) -> np.ndarray:
        length = self.get_obs_component_length(iof, component_id)
        arr = np.empty(length, dtype=np.float64)
        if length > 0:
            self._invoke(
                self.lib.letkf_copy_obs_real_component,
                c_int(iof),
                c_int(component_id),
                c_void_p(arr.ctypes.data),
                c_int(length),
                name="letkf_copy_obs_real_component",
            )
        return arr

    def obs_info_deallocate(self, iof: int) -> None:
        self._invoke(self.lib.letkf_obs_info_deallocate, c_int(iof), name="letkf_obs_info_deallocate")

    # ------------------------------------------------------------------
    # state helpers -----------------------------------------------------
    def get_state_dims(self) -> Dict[str, int]:
        nij1 = c_int()
        nlev = c_int()
        nens = c_int()
        nv3d = c_int()
        nv2d = c_int()
        ierr = c_int()
        self.lib.letkf_get_state_dims(
            ctypes.byref(nij1),
            ctypes.byref(nlev),
            ctypes.byref(nens),
            ctypes.byref(nv3d),
            ctypes.byref(nv2d),
            ctypes.byref(ierr),
        )
        self._check(ierr.value, "letkf_get_state_dims")
        return {
            "nij1": nij1.value,
            "nlev": nlev.value,
            "nens": nens.value,
            "nv3d": nv3d.value,
            "nv2d": nv2d.value,
        }

    def get_state_array(self, array_id: int, is_analysis: bool) -> np.ndarray:
        dims = self.get_state_dims()
        if array_id == self.STATE_3D:
            shape = (dims["nij1"], dims["nlev"], dims["nens"], dims["nv3d"])
        elif array_id == self.STATE_2D:
            nv2d = dims["nv2d"]
            if nv2d == 0:
                return np.empty((dims["nij1"], dims["nens"], 0), dtype=np.float64)
            shape = (dims["nij1"], dims["nens"], nv2d)
        else:
            raise ValueError("Unknown state array id")
        size = int(np.prod(shape))
        arr = np.empty(size, dtype=np.float64, order="F")
        self._invoke(
            self.lib.letkf_copy_state_real,
            c_int(array_id),
            c_int(1 if is_analysis else 0),
            c_void_p(arr.ctypes.data),
            c_int(size),
            c_int(0),
            name="letkf_copy_state_real",
        )
        return arr.reshape(shape, order="F")

    def set_state_array(self, array_id: int, is_analysis: bool, data: np.ndarray) -> None:
        if data.dtype != np.float64:
            raise TypeError("data must be float64")
        arr = np.asfortranarray(data)
        size = arr.size
        self._invoke(
            self.lib.letkf_copy_state_real,
            c_int(array_id),
            c_int(1 if is_analysis else 0),
            c_void_p(arr.ctypes.data),
            c_int(size),
            c_int(1),
            name="letkf_copy_state_real",
        )

    # ------------------------------------------------------------------
    # composite workflow ------------------------------------------------
    def run_letkf(self) -> None:
        info = self.get_common_nml_info()
        mem_required = info.member + (2 if info.det_run else 1)
        self.set_mem_node_proc(mem_required)
        self.set_scalelib("LETKF")

        rank_info = self.get_mpi_rank_info()
        if rank_info.use:
            self.set_common_scale()
            self.set_common_mpi_scale()
            self.set_common_obs_scale()
            self.mtx_setup(info.member)
            self.read_obs_all_mpi()
            if info.obsda_in:
                self.get_nobs_da_mpi()
            else:
                self.set_nobs_extern(0)
            self.obsope_cal()
            self.set_letkf_obs()
            self.dump_letkf_obs_state()
            self.set_common_mpi_grid()
            self.allocate_state_arrays()
            self.read_ens_mpi()
            self.adjust_det_run()
            self.dump_letkf_gues_state()
            if info.departure_stat and info.log_level >= 1:
                self.write_ensmean(calced=False, monit_step=1)
            else:
                self.write_ensmean(calced=False)
            if info.gues_sprd_out:
                self.write_gues_sprd()
            self.das_letkf()
            self.dump_letkf_analysis_state()
            dims = self.get_state_dims()
            self.ensmean_grd(info.member, dims["nens"], dims["nij1"])
            if info.anal_sprd_out:
                self.write_anal_sprd()
            if info.departure_stat and info.log_level >= 1:
                self.write_ens_mpi(monit_step=2)
            else:
                self.write_ens_mpi()

            # Clean observation buffers
            for iof in range(1, self.get_obs_in_num() + 1):
                self.obs_info_deallocate(iof)
            self.unset_common_mpi_scale()

        self.unset_scalelib()
        self.mpi_timer("FINALIZE", 1)

    # ------------------------------------------------------------------
    # internal utilities ------------------------------------------------
    def _define_prototypes(self) -> None:
        ci_p = ctypes.POINTER(c_int)
        self.lib.letkf_initialize_mpi_scale.argtypes = [ci_p]
        self.lib.letkf_finalize_mpi_scale.argtypes = [ci_p]
        self.lib.letkf_get_common_nml_info.argtypes = [ci_p, ci_p, ci_p, ci_p, ci_p, ci_p, ci_p, ci_p]
        self.lib.letkf_get_mpi_rank_info.argtypes = [ci_p, ci_p, ci_p]
        self.lib.letkf_mpi_timer.argtypes = [ctypes.c_char_p, c_int, c_int, ci_p]
        self.lib.letkf_set_mem_node_proc.argtypes = [c_int, ci_p]
        self.lib.letkf_set_scalelib.argtypes = [ctypes.c_char_p, ci_p]
        self.lib.letkf_unset_scalelib.argtypes = [ci_p]
        self.lib.letkf_set_common_scale.argtypes = [ci_p]
        self.lib.letkf_set_common_mpi_scale.argtypes = [ci_p]
        self.lib.letkf_unset_common_mpi_scale.argtypes = [ci_p]
        self.lib.letkf_set_common_obs_scale.argtypes = [ci_p]
        self.lib.letkf_mtx_setup.argtypes = [c_int, ci_p]
        self.lib.letkf_read_obs_all_mpi.argtypes = [ci_p]
        self.lib.letkf_get_obs_in_num.argtypes = [ci_p]
        self.lib.letkf_get_obs_nobs.argtypes = [c_int, ci_p, ci_p]
        self.lib.letkf_get_obs_component_length.argtypes = [c_int, c_int, ci_p, ci_p]
        self.lib.letkf_copy_obs_int_component.argtypes = [c_int, c_int, c_void_p, c_int, ci_p]
        self.lib.letkf_copy_obs_real_component.argtypes = [c_int, c_int, c_void_p, c_int, ci_p]
        self.lib.letkf_obs_info_deallocate.argtypes = [c_int, ci_p]
        self.lib.letkf_get_nobs_da_mpi.argtypes = [ci_p, ci_p]
        self.lib.letkf_set_nobs_extern.argtypes = [c_int, ci_p]
        self.lib.letkf_obsope_cal.argtypes = [ci_p]
        self.lib.letkf_set_letkf_obs.argtypes = [ci_p]
        self.lib.letkf_dump_letkf_obs_state.argtypes = [ci_p]
        self.lib.letkf_dump_letkf_gues_state.argtypes = [ci_p]
        self.lib.letkf_set_common_mpi_grid.argtypes = [ci_p]
        self.lib.letkf_allocate_state_arrays.argtypes = [ci_p]
        self.lib.letkf_read_ens_mpi.argtypes = [ci_p]
        self.lib.letkf_adjust_det_run.argtypes = [ci_p]
        self.lib.letkf_write_enssprd.argtypes = [ci_p]
        self.lib.letkf_write_anal_sprd.argtypes = [ci_p]
        self.lib.letkf_das_letkf.argtypes = [ci_p]
        self.lib.letkf_dump_letkf_analysis_state.argtypes = [ci_p]
        self.lib.letkf_ensmean_grd.argtypes = [c_int, c_int, c_int, ci_p]
        self.lib.letkf_write_ens_mpi.argtypes = [c_int, ci_p]
        self.lib.letkf_write_ensmean.argtypes = [c_int, c_int, ci_p]
        self.lib.letkf_get_state_dims.argtypes = [ci_p, ci_p, ci_p, ci_p, ci_p, ci_p]
        self.lib.letkf_copy_state_real.argtypes = [c_int, c_int, c_void_p, c_int, c_int, ci_p]

    def _invoke(self, func, *args, name: Optional[str] = None) -> None:
        ierr = c_int(0)
        func(*args, ctypes.byref(ierr))
        self._check(ierr.value, name or func.__name__)

    @staticmethod
    def _check(code: int, name: str) -> None:
        if code != 0:
            raise RuntimeError(f"{name} returned error code {code}")


__all__ = ["LetkfAPI", "CommonNMLInfo", "RankInfo"]
