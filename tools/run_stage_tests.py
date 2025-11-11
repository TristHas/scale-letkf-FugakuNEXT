from pathlib import Path
import tools.letkf_core as letkf_core_module
from tools.das_replay import test_obs_local_before_from_global
from tools.letkf_core import LetkfCoreIdentifier, test_letkf_core, test_letkf_core_from_global
from tools.load_das_letkf import list_das_calls
from tools.obs_local import ObsLocalIdentifier, test_local_obs, test_local_obs_from_global
from tools.postproc import PostprocIdentifier, test_postproc, test_postproc_from_global


def run_stage_tests(
      dump_root: Path,
      *,
      pe_tags: list[str] | None = None,
      members: list[str] | None = None,
      call_ids: list[int] | None = None,
      skip_replay_check: bool = False,
    ) -> None:
  dump_root = dump_root.resolve()
  if pe_tags is None: pe_tags = _discover_pe_tags(dump_root)
  if members is None: members = _discover_members(dump_root)
  if call_ids is None: call_ids = list_das_calls(dump_root)

  for pe_tag in pe_tags:
      for member in members:
          print(f"\n==> PE {pe_tag} / member {member} ({len(call_ids)} calls)")
          if not skip_replay_check:
              replay_errs = test_obs_local_before_from_global(dump_root, pe_tag, member, call_ids)
              max_err = max(replay_errs.values(), default=0.0)
              print(f"    [obs_local.before/global] max abs error={max_err:.3e}")

          _run_obs_local_suite(dump_root, pe_tag, member, call_ids)
          _run_letkf_core_suite(dump_root, pe_tag, member, call_ids)
          _run_postproc_suite(dump_root, pe_tag, member, call_ids)


def _run_stage(label, func, identifier):
  errors = func(identifier)
  if errors:
      summary = ", ".join(f"{k}={v:.3e}" for k, v in sorted(errors.items()))
  else:
      summary = "ok"
  print(f"    [{label:>18}] call {identifier.call_id:>6}: {summary}")


def _discover_pe_tags(dump_root: Path) -> list[str]:
  tags = {path.name.split("_", 1)[1].split(".", 1)[0]
          for path in (dump_root / "grid").glob("rig1_pe*.mem*.bin")}
  if not tags:
      raise RuntimeError(f"No rig1_* files found in {dump_root / 'grid'}")
  return sorted(tags)


def _discover_members(dump_root: Path) -> list[str]:
  members = {path.name.split(".")[1] for path in (dump_root / "gues3d").glob("gues3d_pe*.mem*.bin")}
  if not members:
      raise RuntimeError(f"No gues3d files found in {dump_root / 'gues3d'}")
  return sorted(members)


def _run_obs_local_suite(dump_root: Path, pe_tag: str, member: str, call_ids: list[int]) -> None:
  for call_id in call_ids:
      common = {"dump_dir": dump_root, "call_id": call_id, "pe_tag": pe_tag, "member": member}
      _run_stage("obs_local/local", test_local_obs, ObsLocalIdentifier(**common))
      _run_stage("obs_local/global", test_local_obs_from_global, ObsLocalIdentifier(**common))


def _run_letkf_core_suite(dump_root: Path, pe_tag: str, member: str, call_ids: list[int]) -> None:
  _reset_letkf_replay_cache(dump_root, pe_tag, member)
  for call_id in call_ids:
      common = {"dump_dir": dump_root, "call_id": call_id, "pe_tag": pe_tag, "member": member}
      _run_stage("letkf_core/local", test_letkf_core, LetkfCoreIdentifier(**common))
      _run_stage("letkf_core/global", test_letkf_core_from_global, LetkfCoreIdentifier(**common))
  _reset_letkf_replay_cache(dump_root, pe_tag, member)


def _run_postproc_suite(dump_root: Path, pe_tag: str, member: str, call_ids: list[int]) -> None:
  _reset_letkf_replay_cache(dump_root, pe_tag, member)
  for call_id in call_ids:
      common = {"dump_dir": dump_root, "call_id": call_id, "pe_tag": pe_tag, "member": member}
      _run_stage("postproc/local", test_postproc, PostprocIdentifier(**common))
      _run_stage("postproc/global", test_postproc_from_global, PostprocIdentifier(**common))
  _reset_letkf_replay_cache(dump_root, pe_tag, member)


def _reset_letkf_replay_cache(dump_root: Path, pe_tag: str, member: str) -> None:
  key = (str(dump_root), pe_tag, member)
  letkf_core_module._REPLAY_CACHE.pop(key, None)

if __name__ == "__main__":
    # --- configure & run ---
    run_stage_tests(
      dump_root=Path("result/SC23/20210730060030/letkf_dump"),
      pe_tags=["pe000000"],      # None to auto-discover all ranks
      members=["mem0001"],       # None to auto-discover
      call_ids=[1],              # None to run every call
      skip_replay_check=True,    # False to compare replayed vs Fortran before-state
    )