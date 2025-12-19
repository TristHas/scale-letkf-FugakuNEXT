MODULE letkf_dump
  use iso_fortran_env, only: int32, error_unit, int64
  use scale_precision, only: RP
  use common
  use common_nml
  use common_mpi
  use common_mpi_scale
  use common_scale, only: nlon, nlat, nlev
  use common_obs_scale, only: obs_da_value, obsda_sort, obs_info, max_obs_info_meta, obs
  use letkf_obs, only: obsda, obsgrd, nctype, hori_loc_ctype, vert_loc_ctype, nobstotal
  implicit none
  private

  character(len=*), parameter :: dump_suffix_ext = '.bin'
  character(len=*), parameter :: metadata_ext = '.txt'
  character(len=*), parameter :: obs_stage_after_obsope = 'obsda_after_obsope'
  character(len=*), parameter :: obs_stage_after_set = 'obsda_after_set_letkf'
  character(len=*), parameter :: das_stage_root = 'das_letkf'
  character(len=*), parameter :: state_meta_subdir = 'state_meta'
  integer, parameter :: das_dump_rank = -1
  integer(int64), parameter :: das_dump_max_calls = 2000_int64

  character(len=filelenmax), save :: das_base_dir = ''
  logical, save :: das_base_ready = .false.
  logical, save :: das_dump_banner_printed = .false.
  integer, parameter :: das_trace_max = 20
  integer, save :: das_trace_count = 0
  character(len=filelenmax), save :: obsop_cal_base = ''
  logical, save :: obsop_cal_ready = .false.

  public :: dump_letkf_obs_after_obsope
  public :: dump_letkf_obs_after_set
  public :: dump_letkf_gues_state
  public :: dump_letkf_analysis_state
  public :: dump_letkf_raw_obs
  public :: dump_letkf_obsgrd
  public :: dump_letkf_grid_indices
  public :: dump_letkf_localization_tables
  public :: dump_letkf_obs_nosort_coords
  public :: dump_das_obs_local_before
  public :: dump_das_obs_local_after
  public :: dump_das_letkf_core_before
  public :: dump_das_letkf_core_after
  public :: dump_das_postproc_before
  public :: dump_das_postproc_after
  public :: das_dump_enabled
  public :: prepare_das_dump_base
  public :: dump_obsop_cal_state
  public :: dump_obsop_cal_slot

CONTAINS

  SUBROUTINE dump_letkf_obs_after_obsope()
    call dump_obsda_stage(obs_stage_after_obsope, obsda)
  END SUBROUTINE dump_letkf_obs_after_obsope

  SUBROUTINE dump_letkf_obs_after_set()
    call dump_obsda_stage(obs_stage_after_set, obsda_sort)
  END SUBROUTINE dump_letkf_obs_after_set

  SUBROUTINE dump_obsda_stage(stage_tag, obs_data)
    character(len=*), intent(in) :: stage_tag
    type(obs_da_value), intent(in) :: obs_data

    if (.not. LETKF_INPUT_DUMP) return
    if (.not. allocated(obs_data%set)) return
    call export_obsda_dump(trim_dir(LETKF_INPUT_DUMP_DIR), stage_tag, obs_data)
  END SUBROUTINE dump_obsda_stage

  SUBROUTINE dump_letkf_gues_state(gues3d, gues2d)
    real(r_size), intent(in) :: gues3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: gues2d(nij1,nens,nv2d)

    if (.not. LETKF_INPUT_DUMP) return
    call export_state_dump(trim_dir(LETKF_INPUT_DUMP_DIR), 'gues', gues3d, gues2d)
  END SUBROUTINE dump_letkf_gues_state

  SUBROUTINE dump_letkf_analysis_state(anal3d, anal2d, prefix)
    real(r_size), intent(in) :: anal3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: anal2d(nij1,nens,nv2d)
    character(len=*), intent(in), optional :: prefix
    character(len=32) :: use_prefix
    use_prefix = 'anal'
    if (present(prefix)) use_prefix = prefix

    if (.not. LETKF_INPUT_DUMP) return
    call export_state_dump(trim_dir(LETKF_INPUT_DUMP_DIR), use_prefix, anal3d, anal2d)
  END SUBROUTINE dump_letkf_analysis_state

  SUBROUTINE dump_letkf_raw_obs(obs_in)
    type(obs_info), intent(in) :: obs_in(:)

    integer :: iof, ierr
    character(len=filelenmax) :: base_dir
    character(len=filelenmax) :: dir_stage
    character(len=filelenmax) :: obs_dir
    character(len=16) :: obs_tag
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag

    if (.not. LETKF_INPUT_DUMP) return

    base_dir = trim_dir(LETKF_INPUT_DUMP_DIR)
    dir_stage = append_dir(base_dir, 'obs_raw')
    call ensure_directory(dir_stage)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    do iof = 1, size(obs_in)
      write(obs_tag,'(A3,I4.4)') 'obs', iof
      obs_dir = append_dir(dir_stage, trim(obs_tag))
      call ensure_directory(obs_dir)
      if (myrank == 0) then
        call write_obs_record(obs_dir, obs_in(iof), iof, domain_tag, ensemble_tag)
      end if
    end do

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
  END SUBROUTINE dump_letkf_raw_obs

  SUBROUTINE dump_letkf_obsgrd()
    integer :: ictype, ierr
    character(len=filelenmax) :: base_dir
    character(len=filelenmax) :: dir_stage
    character(len=filelenmax) :: ctype_dir
    character(len=12) :: ctype_tag
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag

    if (.not. LETKF_INPUT_DUMP) return
    if (.not. allocated(obsgrd)) return
    if (nctype <= 0) return

    base_dir = trim_dir(LETKF_INPUT_DUMP_DIR)
    dir_stage = append_dir(base_dir, 'obsgrd')
    call ensure_directory(dir_stage)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    call write_obsgrd_summary(dir_stage, domain_tag, ensemble_tag)

    do ictype = 1, nctype
      write(ctype_tag,'(A5,I5.5)') 'type_', ictype
      ctype_dir = append_dir(dir_stage, trim(ctype_tag))
      call ensure_directory(ctype_dir)
      call write_obsgrd_meta(ctype_dir, ictype, domain_tag, ensemble_tag)
      call dump_obsgrd_arrays(ctype_dir, ictype, domain_tag, ensemble_tag)
    end do

    call MPI_Barrier(MPI_COMM_WORLD, ierr)
  END SUBROUTINE dump_letkf_obsgrd

  SUBROUTINE dump_letkf_grid_indices(rig1, rjg1, topo1, hgt1)
    real(r_size), intent(in) :: rig1(:)
    real(r_size), intent(in) :: rjg1(:)
    real(r_size), intent(in), optional :: topo1(:)
    real(r_size), intent(in), optional :: hgt1(:,:)
    character(len=filelenmax) :: base_dir
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=filelenmax) :: file_path

    if (.not. LETKF_INPUT_DUMP) return

    base_dir = append_dir(trim_dir(LETKF_INPUT_DUMP_DIR), 'grid')
    call ensure_directory(base_dir)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    file_path = build_rank_filename(base_dir, 'rig1', domain_tag, ensemble_tag)
    call write_real_vector(file_path, rig1)

    file_path = build_rank_filename(base_dir, 'rjg1', domain_tag, ensemble_tag)
    call write_real_vector(file_path, rjg1)
    if (present(topo1)) then
      file_path = build_rank_filename(base_dir, 'topo1', domain_tag, ensemble_tag)
      call write_real_vector(file_path, topo1)
    end if
    if (present(hgt1)) then
      file_path = build_rank_filename(base_dir, 'hgt1', domain_tag, ensemble_tag)
      call write_real_matrix(file_path, hgt1)
    end if
  END SUBROUTINE dump_letkf_grid_indices

  SUBROUTINE dump_letkf_localization_tables(var_local, var_local_n2nc, var_local_n2n, uid_obs_varlocal, n_merge, ic_merge, elm_u_ctype, typ_ctype)
    real(r_size), intent(in) :: var_local(:,:)
    integer, intent(in) :: var_local_n2nc(:)
    integer, intent(in) :: var_local_n2n(:)
    integer, intent(in) :: uid_obs_varlocal(:)
    integer, intent(in) :: n_merge(:)
    integer, intent(in) :: ic_merge(:,:)
    integer, intent(in) :: elm_u_ctype(:)
    integer, intent(in) :: typ_ctype(:)
    character(len=filelenmax) :: base_dir
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag

    if (.not. LETKF_INPUT_DUMP) return

    base_dir = append_dir(trim_dir(LETKF_INPUT_DUMP_DIR), 'localization')
    call ensure_directory(base_dir)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    call write_real_matrix(trim(base_dir)//'/'//'var_local_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, var_local)
    call write_integer_vector(trim(base_dir)//'/'//'var_local_n2nc_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, var_local_n2nc)
    call write_integer_vector(trim(base_dir)//'/'//'var_local_n2n_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, var_local_n2n)
    call write_integer_vector(trim(base_dir)//'/'//'uid_obs_varlocal_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, uid_obs_varlocal)
    call write_integer_vector(trim(base_dir)//'/'//'n_merge_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, n_merge)
    call write_integer_matrix(trim(base_dir)//'/'//'ic_merge_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, ic_merge)
    call write_integer_vector(trim(base_dir)//'/'//'elm_u_ctype_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, elm_u_ctype)
    call write_integer_vector(trim(base_dir)//'/'//'typ_ctype_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext, typ_ctype)
  END SUBROUTINE dump_letkf_localization_tables

  SUBROUTINE dump_obsop_cal_state(iter, mem, islot, v3dg, v2dg, mv3d_slot, slope3d_slot)
    integer, intent(in) :: iter
    integer, intent(in) :: mem
    integer, intent(in) :: islot
    real(r_size), intent(in) :: v3dg(:,:,:,:)
    real(r_size), intent(in) :: v2dg(:,:,:)
    real(r_size), intent(in), optional :: mv3d_slot(:,:,:,:)
    real(r_size), intent(in), optional :: slope3d_slot(:,:)

    character(len=filelenmax) :: dir_stage
    character(len=filelenmax) :: dir_root
    character(len=filelenmax) :: file_path
    character(len=32) :: stage_tag
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag

    if (.not. LETKF_INPUT_DUMP) return
    call ensure_obsop_cal_dir()

    stage_tag = obsop_stage_tag(iter, islot)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix_for_mem(mem)

    dir_root = append_dir(obsop_cal_base, 'state')
    dir_root = append_dir(dir_root, domain_tag)
    dir_root = append_dir(dir_root, ensemble_tag)
    dir_stage = append_dir(dir_root, stage_tag)
    call ensure_directory(dir_stage)

    file_path = append_dir(dir_stage, 'v3dg'//dump_suffix_ext)
    call write_real4d(file_path, real(v3dg, RP))

    file_path = append_dir(dir_stage, 'v2dg'//dump_suffix_ext)
    call write_real3d(file_path, real(v2dg, RP))

    if (present(mv3d_slot)) then
      file_path = append_dir(dir_stage, 'mv3dg'//dump_suffix_ext)
      call write_real4d(file_path, real(mv3d_slot, RP))
    end if

    if (present(slope3d_slot)) then
      file_path = append_dir(dir_stage, 'slope3dg'//dump_suffix_ext)
      call write_real_matrix(file_path, slope3d_slot)
    end if

    call write_stage_metadata(dir_stage, iter, islot, mem)
  END SUBROUTINE dump_obsop_cal_state

  SUBROUTINE dump_obsop_cal_slot(iter, mem, islot, obs_sets, obs_idxs, obs_nn, obs_elms, obs_typs, &
                                 obs_lon, obs_lat, obs_lev, obs_meta, obs_ri_global, obs_rj_global, &
                                 obs_ril, obs_rjl, obs_rkz, obs_qc, obs_n1, obs_n2)
    integer, intent(in) :: iter
    integer, intent(in) :: mem
    integer, intent(in) :: islot
    integer, intent(in) :: obs_sets(:)
    integer, intent(in) :: obs_idxs(:)
    integer, intent(in) :: obs_nn(:)
    integer, intent(in) :: obs_elms(:)
    integer, intent(in) :: obs_typs(:)
    real(r_size), intent(in) :: obs_lon(:)
    real(r_size), intent(in) :: obs_lat(:)
    real(r_size), intent(in) :: obs_lev(:)
    real(r_size), intent(in) :: obs_meta(:,:)
    real(r_size), intent(in) :: obs_ri_global(:)
    real(r_size), intent(in) :: obs_rj_global(:)
    real(r_size), intent(in) :: obs_ril(:)
    real(r_size), intent(in) :: obs_rjl(:)
    real(r_size), intent(in) :: obs_rkz(:)
    integer, intent(in) :: obs_qc(:)
    integer, intent(in) :: obs_n1
    integer, intent(in) :: obs_n2

    integer :: nobs
    character(len=filelenmax) :: dir_stage
    character(len=filelenmax) :: dir_root
    character(len=filelenmax) :: file_path
    character(len=32) :: stage_tag
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag

    if (.not. LETKF_INPUT_DUMP) return
    nobs = size(obs_sets)
    if (nobs <= 0) return
    call ensure_obsop_cal_dir()

    stage_tag = obsop_stage_tag(iter, islot)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix_for_mem(mem)

    dir_root = append_dir(obsop_cal_base, 'obs')
    dir_root = append_dir(dir_root, domain_tag)
    dir_root = append_dir(dir_root, ensemble_tag)
    dir_stage = append_dir(dir_root, stage_tag)
    call ensure_directory(dir_stage)

    file_path = append_dir(dir_stage, 'set'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_sets)

    file_path = append_dir(dir_stage, 'idx'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_idxs)

    file_path = append_dir(dir_stage, 'nn'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_nn)

    file_path = append_dir(dir_stage, 'elm'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_elms)

    file_path = append_dir(dir_stage, 'typ'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_typs)

    file_path = append_dir(dir_stage, 'qc'//dump_suffix_ext)
    call write_integer_vector(file_path, obs_qc)

    file_path = append_dir(dir_stage, 'lon'//dump_suffix_ext)
    call write_real_vector(file_path, obs_lon)

    file_path = append_dir(dir_stage, 'lat'//dump_suffix_ext)
    call write_real_vector(file_path, obs_lat)

    file_path = append_dir(dir_stage, 'lev'//dump_suffix_ext)
    call write_real_vector(file_path, obs_lev)

    file_path = append_dir(dir_stage, 'ri_global'//dump_suffix_ext)
    call write_real_vector(file_path, obs_ri_global)

    file_path = append_dir(dir_stage, 'rj_global'//dump_suffix_ext)
    call write_real_vector(file_path, obs_rj_global)

    file_path = append_dir(dir_stage, 'ril'//dump_suffix_ext)
    call write_real_vector(file_path, obs_ril)

    file_path = append_dir(dir_stage, 'rjl'//dump_suffix_ext)
    call write_real_vector(file_path, obs_rjl)

    file_path = append_dir(dir_stage, 'rkz'//dump_suffix_ext)
    call write_real_vector(file_path, obs_rkz)

    file_path = append_dir(dir_stage, 'meta'//dump_suffix_ext)
    call write_real_matrix(file_path, obs_meta)

    call write_stage_metadata(dir_stage, iter, islot, mem, nobs, obs_n1, obs_n2)
  END SUBROUTINE dump_obsop_cal_slot

  SUBROUTINE dump_letkf_obs_nosort_coords()
    integer :: n
    integer :: set_id, idx_id
    integer :: nobs_total
    character(len=filelenmax) :: base_dir
    character(len=filelenmax) :: stage_dir
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=filelenmax) :: file_path
    real(r_size), allocatable :: ri_vals(:)
    real(r_size), allocatable :: rj_vals(:)

    if (.not. LETKF_INPUT_DUMP) return
    if (.not. allocated(obs)) return
    if (obsda_sort%nobs <= 0) return

    nobs_total = obsda_sort%nobs
    allocate(ri_vals(nobs_total))
    allocate(rj_vals(nobs_total))

    do n = 1, nobs_total
      set_id = obsda_sort%set(n)
      idx_id = obsda_sort%idx(n)
      if (set_id >= 1 .and. set_id <= size(obs)) then
        if (allocated(obs(set_id)%ri) .and. idx_id >= 1 .and. idx_id <= size(obs(set_id)%ri)) then
          ri_vals(n) = obs(set_id)%ri(idx_id)
        else
          ri_vals(n) = 0.0_r_size
        end if
        if (allocated(obs(set_id)%rj) .and. idx_id >= 1 .and. idx_id <= size(obs(set_id)%rj)) then
          rj_vals(n) = obs(set_id)%rj(idx_id)
        else
          rj_vals(n) = 0.0_r_size
        end if
      else
        ri_vals(n) = 0.0_r_size
        rj_vals(n) = 0.0_r_size
      end if
    end do

    base_dir = trim_dir(LETKF_INPUT_DUMP_DIR)
    stage_dir = append_dir(base_dir, obs_stage_after_set)
    call ensure_directory(stage_dir)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    file_path = build_rank_filename(stage_dir, 'obsdanosort_ri', domain_tag, ensemble_tag)
    call write_real_vector(file_path, ri_vals)
    file_path = build_rank_filename(stage_dir, 'obsdanosort_rj', domain_tag, ensemble_tag)
    call write_real_vector(file_path, rj_vals)

    deallocate(ri_vals)
    deallocate(rj_vals)
  END SUBROUTINE dump_letkf_obs_nosort_coords

  SUBROUTINE dump_das_obs_local_before(call_id, ilev, ij, nvar_global, n2nc, n2n, kind, ri, rj, rlev, rz, search_q0)
    integer(int64), intent(in) :: call_id
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    character(len=*), intent(in) :: kind
    real(r_size), intent(in) :: ri, rj, rlev, rz
    integer, intent(in) :: search_q0(:)
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(20)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('obs_local', 'before', dir_phase)
    call das_dump_trace('obs_local','before',call_id)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'obs_local')
    call append_meta(meta_lines, nmeta, 'phase', 'before')
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_real(meta_lines, nmeta, 'ri', ri)
    call append_meta_real(meta_lines, nmeta, 'rj', rj)
    call append_meta_real(meta_lines, nmeta, 'rlev', rlev)
    call append_meta_real(meta_lines, nmeta, 'rz', rz)
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'search_q0', call_id, domain_tag, ensemble_tag)
    call write_integer_vector(file_path, search_q0)
  END SUBROUTINE dump_das_obs_local_before

  SUBROUTINE dump_das_obs_local_after(call_id, ilev, ij, nvar_global, n2nc, n2n, kind, nobsl, hdxf, rdiag, rloc, dep, nobsl_t, cutd_t, search_q0)
    integer(int64), intent(in) :: call_id
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    character(len=*), intent(in) :: kind
    integer, intent(in) :: nobsl
    real(r_size), intent(in) :: hdxf(:,:)
    real(r_size), intent(in) :: rdiag(:)
    real(r_size), intent(in) :: rloc(:)
    real(r_size), intent(in) :: dep(:)
    integer, intent(in) :: nobsl_t(:,:)
    real(r_size), intent(in) :: cutd_t(:,:)
    integer, intent(in) :: search_q0(:)
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(24)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('obs_local', 'after', dir_phase)
    call das_dump_trace('obs_local','after',call_id)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'obs_local')
    call append_meta(meta_lines, nmeta, 'phase', 'after')
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_int(meta_lines, nmeta, 'nobsl', int(nobsl, int64))
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'hdxf', call_id, domain_tag, ensemble_tag)
    call write_real_matrix_prefix(file_path, hdxf, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'rdiag', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, rdiag, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'rloc', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, rloc, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'dep', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, dep, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'search_q0', call_id, domain_tag, ensemble_tag)
    call write_integer_vector(file_path, search_q0)
    file_path = build_das_rank_filename(dir_phase, 'nobsl_t', call_id, domain_tag, ensemble_tag)
    call write_integer2d(file_path, nobsl_t)
    file_path = build_das_rank_filename(dir_phase, 'cutd_t', call_id, domain_tag, ensemble_tag)
    call write_real_matrix(file_path, cutd_t)
  END SUBROUTINE dump_das_obs_local_after

  SUBROUTINE dump_das_letkf_core_before(call_id, kind, ilev, ij, nvar_global, n2nc, n2n, nobsl, nobstotal_all, parm_infl, hdxf, rdiag, rloc, dep, rdiag_wloc_flag, infl_update_flag)
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: kind
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    integer, intent(in) :: nobsl, nobstotal_all
    real(r_size), intent(in) :: parm_infl
    real(r_size), intent(in) :: hdxf(:,:)
    real(r_size), intent(in) :: rdiag(:)
    real(r_size), intent(in) :: rloc(:)
    real(r_size), intent(in) :: dep(:)
    logical, intent(in) :: rdiag_wloc_flag
    logical, intent(in) :: infl_update_flag
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(32)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('letkf_core', 'before', dir_phase)
    call das_dump_trace('letkf_core','before',call_id)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'letkf_core')
    call append_meta(meta_lines, nmeta, 'phase', 'before')
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta_int(meta_lines, nmeta, 'nobsl', int(nobsl, int64))
    call append_meta_int(meta_lines, nmeta, 'nobstotal', int(nobstotal_all, int64))
    call append_meta_real(meta_lines, nmeta, 'parm_infl', parm_infl)
    call append_meta_logical(meta_lines, nmeta, 'rdiag_wloc', rdiag_wloc_flag)
    call append_meta_logical(meta_lines, nmeta, 'infl_update', infl_update_flag)
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'hdxf', call_id, domain_tag, ensemble_tag)
    call write_real_matrix_prefix(file_path, hdxf, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'rdiag', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, rdiag, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'rloc', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, rloc, nobsl)
    file_path = build_das_rank_filename(dir_phase, 'dep', call_id, domain_tag, ensemble_tag)
    call write_real_vector_prefix(file_path, dep, nobsl)
  END SUBROUTINE dump_das_letkf_core_before

  SUBROUTINE dump_das_letkf_core_after(call_id, kind, ilev, ij, nvar_global, n2nc, n2n, nobsl, parm_infl, trans, transm, pa, parm_updated, transmd)
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: kind
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    integer, intent(in) :: nobsl
    real(r_size), intent(in) :: parm_infl
    real(r_size), intent(in) :: trans(:,:)
    real(r_size), intent(in) :: transm(:)
    real(r_size), intent(in) :: pa(:,:)
    real(r_size), intent(in) :: parm_updated
    real(r_size), intent(in), optional :: transmd(:)
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(28)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('letkf_core', 'after', dir_phase)
    call das_dump_trace('letkf_core','after',call_id)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'letkf_core')
    call append_meta(meta_lines, nmeta, 'phase', 'after')
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta_int(meta_lines, nmeta, 'nobsl', int(nobsl, int64))
    call append_meta_real(meta_lines, nmeta, 'parm_infl_post', parm_infl)
    call append_meta_real(meta_lines, nmeta, 'parm_updated', parm_updated)
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'trans', call_id, domain_tag, ensemble_tag)
    call write_real_matrix(file_path, trans)
    file_path = build_das_rank_filename(dir_phase, 'transm', call_id, domain_tag, ensemble_tag)
    call write_real_vector(file_path, transm)
    file_path = build_das_rank_filename(dir_phase, 'pa', call_id, domain_tag, ensemble_tag)
    call write_real_matrix(file_path, pa)
    if (present(transmd)) then
      file_path = build_das_rank_filename(dir_phase, 'transmd', call_id, domain_tag, ensemble_tag)
      call write_real_vector(file_path, transmd)
    end if
  END SUBROUTINE dump_das_letkf_core_after

  SUBROUTINE dump_das_postproc_before(call_id, kind, ilev, ij, nvar_global, n2nc, n2n, beta, parm_val, relax_alpha, relax_alpha_spread, relax_spread_out, relax_to_inflated_prior, det_run, gues_mean, gues_members, trans, transm)
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: kind
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    real(r_size), intent(in) :: beta
    real(r_size), intent(in) :: parm_val
    real(r_size), intent(in) :: relax_alpha, relax_alpha_spread
    logical, intent(in) :: relax_spread_out, relax_to_inflated_prior, det_run
    real(r_size), intent(in) :: gues_mean
    real(r_size), intent(in) :: gues_members(:)
    real(r_size), intent(in) :: trans(:,:)
    real(r_size), intent(in) :: transm(:)
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(32)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('postproc', 'before', dir_phase)
    call das_dump_trace('postproc','before',call_id)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'postproc')
    call append_meta(meta_lines, nmeta, 'phase', 'before')
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta_real(meta_lines, nmeta, 'beta', beta)
    call append_meta_real(meta_lines, nmeta, 'parm', parm_val)
    call append_meta_real(meta_lines, nmeta, 'relax_alpha', relax_alpha)
    call append_meta_real(meta_lines, nmeta, 'relax_alpha_spread', relax_alpha_spread)
    call append_meta_logical(meta_lines, nmeta, 'relax_spread_out', relax_spread_out)
    call append_meta_logical(meta_lines, nmeta, 'relax_to_inflated_prior', relax_to_inflated_prior)
    call append_meta_logical(meta_lines, nmeta, 'det_run', det_run)
    call append_meta_real(meta_lines, nmeta, 'gues_mean', gues_mean)
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'trans', call_id, domain_tag, ensemble_tag)
    call write_real_matrix(file_path, trans)
    file_path = build_das_rank_filename(dir_phase, 'transm', call_id, domain_tag, ensemble_tag)
    call write_real_vector(file_path, transm)
    file_path = build_das_rank_filename(dir_phase, 'gues_members', call_id, domain_tag, ensemble_tag)
    call write_real_vector(file_path, gues_members)
  END SUBROUTINE dump_das_postproc_before

  SUBROUTINE dump_das_postproc_after(call_id, kind, ilev, ij, nvar_global, n2nc, n2n, beta, transrlx, anal_members, q_mean, q_sprd, q_limited, workda_value, workda_present, anal_det)
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: kind
    integer, intent(in) :: ilev, ij, nvar_global, n2nc, n2n
    real(r_size), intent(in) :: beta
    real(r_size), intent(in) :: transrlx(:,:)
    real(r_size), intent(in) :: anal_members(:)
    real(r_size), intent(in) :: q_mean, q_sprd
    logical, intent(in) :: q_limited
    real(r_size), intent(in) :: workda_value
    logical, intent(in) :: workda_present
    real(r_size), intent(in), optional :: anal_det(:)
    character(len=filelenmax) :: dir_phase
    character(len=filelenmax) :: file_path
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=256) :: meta_lines(32)
    integer :: nmeta

    if (.not. das_dump_allow(call_id)) return

    call get_das_phase_dir('postproc', 'after', dir_phase)
    call das_dump_trace('postproc','after',call_id)
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()

    nmeta = 0
    call append_meta(meta_lines, nmeta, 'stage', 'postproc')
    call append_meta(meta_lines, nmeta, 'phase', 'after')
    call append_meta(meta_lines, nmeta, 'kind', trim(kind))
    call append_meta_int(meta_lines, nmeta, 'call_id', call_id)
    call append_meta_int(meta_lines, nmeta, 'ij', int(ij, int64))
    call append_meta_int(meta_lines, nmeta, 'ilev', int(ilev, int64))
    call append_meta_int(meta_lines, nmeta, 'nvar', int(nvar_global, int64))
    call append_meta_int(meta_lines, nmeta, 'n2nc', int(n2nc, int64))
    call append_meta_int(meta_lines, nmeta, 'n2n', int(n2n, int64))
    call append_meta_real(meta_lines, nmeta, 'beta', beta)
    call append_meta_real(meta_lines, nmeta, 'q_mean', q_mean)
    call append_meta_real(meta_lines, nmeta, 'q_sprd', q_sprd)
    call append_meta_logical(meta_lines, nmeta, 'q_limited', q_limited)
    call append_meta_logical(meta_lines, nmeta, 'workda_present', workda_present)
    call append_meta_real(meta_lines, nmeta, 'workda_value', workda_value)
    call write_das_metadata(dir_phase, call_id, meta_lines(1:nmeta))

    file_path = build_das_rank_filename(dir_phase, 'transrlx', call_id, domain_tag, ensemble_tag)
    call write_real_matrix(file_path, transrlx)
    file_path = build_das_rank_filename(dir_phase, 'anal_members', call_id, domain_tag, ensemble_tag)
    call write_real_vector(file_path, anal_members)
    if (present(anal_det)) then
      file_path = build_das_rank_filename(dir_phase, 'anal_det', call_id, domain_tag, ensemble_tag)
      call write_real_vector(file_path, anal_det)
    end if
  END SUBROUTINE dump_das_postproc_after

  SUBROUTINE export_state_dump(base_dir, prefix, state3d, state2d)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    real(r_size), intent(in) :: state3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: state2d(nij1,nens,nv2d)

    real(RP), allocatable :: v3dr(:,:,:,:)
    real(RP), allocatable :: v2dr(:,:,:)
    character(len=filelenmax) :: file3d, file2d
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    integer :: ierr
    character(len=filelenmax) :: dir_local
    character(len=filelenmax) :: dir_3d
    character(len=filelenmax) :: dir_2d
    character(len=filelenmax) :: dir_meta

    dir_local = base_dir
    call ensure_directory(dir_local)
    dir_meta = append_dir(dir_local, state_meta_subdir)
    call ensure_directory(dir_meta)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    call write_state_metadata(dir_meta, domain_tag, ensemble_tag)

    if (nv3d > 0) then
      dir_3d = append_dir(dir_local, trim(prefix)//'3d')
      call ensure_directory(dir_3d)
      allocate(v3dr(nij1,nlev,nens,nv3d))
      v3dr = real(state3d, RP)
      file3d = build_rank_filename(dir_3d, trim(prefix)//'3d', domain_tag, ensemble_tag)
      call write_real4d(file3d, v3dr)
      deallocate(v3dr)
    end if

    if (nv2d > 0) then
      dir_2d = append_dir(dir_local, trim(prefix)//'2d')
      call ensure_directory(dir_2d)
      allocate(v2dr(nij1,nens,nv2d))
      v2dr = real(state2d, RP)
      file2d = build_rank_filename(dir_2d, trim(prefix)//'2d', domain_tag, ensemble_tag)
      call write_real3d(file2d, v2dr)
      deallocate(v2dr)
    end if

    call MPI_Barrier(MPI_COMM_e, ierr)
  END SUBROUTINE export_state_dump

  SUBROUTINE export_obsda_dump(base_dir, stage_tag, obs_data)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: stage_tag
    type(obs_da_value), intent(in) :: obs_data

    character(len=filelenmax) :: dir_stage
    character(len=filelenmax) :: file_name
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    integer :: ierr

    if (.not. allocated(obs_data%set)) return

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    dir_stage = append_dir(base_dir, stage_tag)
    call ensure_directory(dir_stage)

    if (allocated(obs_data%set)) then
      file_name = obsda_component_filename(dir_stage, 'set', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_data%set)
    end if

    if (allocated(obs_data%idx)) then
      file_name = obsda_component_filename(dir_stage, 'idx', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_data%idx)
    end if

    if (allocated(obs_data%key)) then
      file_name = obsda_component_filename(dir_stage, 'key', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_data%key)
    end if

    if (allocated(obs_data%val)) then
      file_name = obsda_component_filename(dir_stage, 'val', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_data%val)
    end if

    if (allocated(obs_data%ensval)) then
      file_name = obsda_component_filename(dir_stage, 'ensval', domain_tag, ensemble_tag)
      call write_real_matrix(file_name, obs_data%ensval)
    end if

    if (allocated(obs_data%epert)) then
      file_name = obsda_component_filename(dir_stage, 'epert', domain_tag, ensemble_tag)
      call write_real_matrix(file_name, obs_data%epert)
    end if

    if (allocated(obs_data%pert)) then
      file_name = obsda_component_filename(dir_stage, 'pert', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_data%pert)
    end if

    if (allocated(obs_data%eqv)) then
      file_name = obsda_component_filename(dir_stage, 'eqv', domain_tag, ensemble_tag)
      call write_real_matrix(file_name, obs_data%eqv)
    end if

    if (allocated(obs_data%qv)) then
      file_name = obsda_component_filename(dir_stage, 'qv', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_data%qv)
    end if

    if (allocated(obs_data%tm)) then
      file_name = obsda_component_filename(dir_stage, 'tm', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_data%tm)
    end if

    if (allocated(obs_data%pm)) then
      file_name = obsda_component_filename(dir_stage, 'pm', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_data%pm)
    end if

    if (allocated(obs_data%qc)) then
      file_name = obsda_component_filename(dir_stage, 'qc', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_data%qc)
    end if

    call MPI_Barrier(MPI_COMM_d, ierr)
  END SUBROUTINE export_obsda_dump

  SUBROUTINE write_obs_record(obs_dir, obs_rec, obs_index, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: obs_dir
    type(obs_info), intent(in) :: obs_rec
    integer, intent(in) :: obs_index
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: file_name
    character(len=memflen+16) :: prefix_base

    prefix_base = trim(obs_dir_tag(obs_index))

    call write_obs_meta(obs_dir, obs_rec, obs_index, domain_tag, ensemble_tag)

    if (allocated(obs_rec%elm)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_elm', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_rec%elm)
    end if
    if (allocated(obs_rec%lon)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_lon', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%lon)
    end if
    if (allocated(obs_rec%lat)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_lat', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%lat)
    end if
    if (allocated(obs_rec%lev)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_lev', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%lev)
    end if
    if (allocated(obs_rec%dat)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_dat', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%dat)
    end if
    if (allocated(obs_rec%err)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_err', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%err)
    end if
    if (allocated(obs_rec%typ)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_typ', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_rec%typ)
    end if
    if (allocated(obs_rec%dif)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_dif', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%dif)
    end if
    if (allocated(obs_rec%ri)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_ri', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%ri)
    end if
    if (allocated(obs_rec%rj)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_rj', domain_tag, ensemble_tag)
      call write_real_vector(file_name, obs_rec%rj)
    end if
    if (allocated(obs_rec%rank)) then
      file_name = build_rank_filename(obs_dir, trim(prefix_base)//'_rank', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obs_rec%rank)
    end if
  END SUBROUTINE write_obs_record

  SUBROUTINE write_obs_meta(obs_dir, obs_rec, obs_index, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: obs_dir
    type(obs_info), intent(in) :: obs_rec
    integer, intent(in) :: obs_index
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: meta_file
    integer :: unit, i

    meta_file = trim(obs_dir)//'/obs_meta_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    open(newunit=unit, file=trim(meta_file), status='replace', action='write')
    write(unit,'(A,I0)') 'obs_index=', obs_index
    write(unit,'(A,I0)') 'nobs=', obs_rec%nobs
    do i = 1, max_obs_info_meta
      write(unit,'(A,I0,A,F24.10)') 'meta(', i, ')=', obs_rec%meta(i)
    end do
    close(unit)
  END SUBROUTINE write_obs_meta

  FUNCTION obs_dir_tag(obs_index) RESULT(tag)
    integer, intent(in) :: obs_index
    character(len=16) :: tag
    write(tag,'("obs",I4.4)') obs_index
  END FUNCTION obs_dir_tag

  SUBROUTINE write_obsgrd_summary(dir_stage, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: dir_stage
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: meta_file
    character(len=filelenmax) :: file_name
    integer :: unit

    meta_file = trim(dir_stage)//'/obsgrd_summary_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    open(newunit=unit, file=trim(meta_file), status='replace', action='write')
    write(unit,'(A,I0)') 'nctype=', nctype
    write(unit,'(A,I0)') 'nobstotal=', nobstotal
    close(unit)

    if (nctype > 0) then
      file_name = build_rank_filename(dir_stage, 'obsgrd_hori_loc', domain_tag, ensemble_tag)
      call write_real_vector(file_name, hori_loc_ctype(1:nctype))
      file_name = build_rank_filename(dir_stage, 'obsgrd_vert_loc', domain_tag, ensemble_tag)
      call write_real_vector(file_name, vert_loc_ctype(1:nctype))
    end if
  END SUBROUTINE write_obsgrd_summary

  SUBROUTINE write_obsgrd_meta(ctype_dir, ictype, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: ctype_dir
    integer, intent(in) :: ictype
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: meta_file
    integer :: unit

    meta_file = trim(ctype_dir)//'/obsgrd_meta_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    open(newunit=unit, file=trim(meta_file), status='replace', action='write')
    write(unit,'(A,I0)') 'ctype=', ictype
    write(unit,'(A,I0)') 'ngrd_i=', obsgrd(ictype)%ngrd_i
    write(unit,'(A,I0)') 'ngrd_j=', obsgrd(ictype)%ngrd_j
    write(unit,'(A,I0)') 'ngrdsch_i=', obsgrd(ictype)%ngrdsch_i
    write(unit,'(A,I0)') 'ngrdsch_j=', obsgrd(ictype)%ngrdsch_j
    write(unit,'(A,I0)') 'ngrdext_i=', obsgrd(ictype)%ngrdext_i
    write(unit,'(A,I0)') 'ngrdext_j=', obsgrd(ictype)%ngrdext_j
    write(unit,'(A,F24.10)') 'grdspc_i=', obsgrd(ictype)%grdspc_i
    write(unit,'(A,F24.10)') 'grdspc_j=', obsgrd(ictype)%grdspc_j
    write(unit,'(A,I0)') 'tot_ext=', obsgrd(ictype)%tot_ext
    close(unit)
  END SUBROUTINE write_obsgrd_meta

  SUBROUTINE dump_obsgrd_arrays(ctype_dir, ictype, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: ctype_dir
    integer, intent(in) :: ictype
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: file_name
    character(len=16) :: base_tag

    write(base_tag,'("type",I4.4)') ictype

    if (allocated(obsgrd(ictype)%n)) then
      file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_n', domain_tag, ensemble_tag)
      call write_integer3d(file_name, obsgrd(ictype)%n)
    end if
    if (allocated(obsgrd(ictype)%ac)) then
      file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_ac', domain_tag, ensemble_tag)
      call write_integer3d(file_name, obsgrd(ictype)%ac)
    end if
    if (allocated(obsgrd(ictype)%tot)) then
      file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_tot', domain_tag, ensemble_tag)
      call write_integer_vector(file_name, obsgrd(ictype)%tot)
    end if
    if (allocated(obsgrd(ictype)%n_ext)) then
      file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_n_ext', domain_tag, ensemble_tag)
      call write_integer2d(file_name, obsgrd(ictype)%n_ext)
    end if
    if (allocated(obsgrd(ictype)%ac_ext)) then
      file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_ac_ext', domain_tag, ensemble_tag)
      call write_integer2d(file_name, obsgrd(ictype)%ac_ext)
    end if
    file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_tot_sub', domain_tag, ensemble_tag)
    call write_integer_vector(file_name, obsgrd(ictype)%tot_sub)
    file_name = build_rank_filename(ctype_dir, trim(base_tag)//'_tot_g', domain_tag, ensemble_tag)
    call write_integer_vector(file_name, obsgrd(ictype)%tot_g)
  END SUBROUTINE dump_obsgrd_arrays

  FUNCTION obsda_component_filename(dir_stage, component_name, domain_tag, ensemble_tag) RESULT(path)
    character(len=*), intent(in) :: dir_stage
    character(len=*), intent(in) :: component_name
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag
    character(len=filelenmax) :: path

    path = trim(dir_stage)//'/obsda_'//trim(adjustl(component_name))//'_'// &
      trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext
  END FUNCTION obsda_component_filename

  SUBROUTINE write_integer_vector(filename, data)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: data(:)
    integer(int32) :: nd, dims(1)
    integer :: unit

    nd = 1_int32
    dims(1) = int(size(data,1), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    if (size(data,1) > 0) write(unit) data
    close(unit)
  END SUBROUTINE write_integer_vector

  SUBROUTINE write_integer_matrix(filename, data)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: data(:,:)
    integer :: unit
    integer :: dims(2)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    dims = shape(data)
    write(unit) 2
    write(unit) dims
    write(unit) data
    close(unit)
  END SUBROUTINE write_integer_matrix

  SUBROUTINE write_real_vector(filename, data)
    character(len=*), intent(in) :: filename
    real(r_size), intent(in) :: data(:)
    integer(int32) :: nd, dims(1)
    integer :: unit

    nd = 1_int32
    dims(1) = int(size(data,1), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    if (size(data,1) > 0) write(unit) data
    close(unit)
  END SUBROUTINE write_real_vector

  SUBROUTINE write_integer2d(filename, data)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: data(:,:)
    integer(int32) :: nd, dims(2)
    integer :: unit

    nd = 2_int32
    dims = int((/size(data,1), size(data,2)/), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    if (size(data,1) > 0 .and. size(data,2) > 0) write(unit) data
    close(unit)
  END SUBROUTINE write_integer2d

  SUBROUTINE write_integer3d(filename, data)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: data(:,:,:)
    integer(int32) :: nd, dims(3)
    integer :: unit

    nd = 3_int32
    dims = int((/size(data,1), size(data,2), size(data,3)/), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    if (size(data,1) > 0 .and. size(data,2) > 0 .and. size(data,3) > 0) write(unit) data
    close(unit)
  END SUBROUTINE write_integer3d

  SUBROUTINE write_real_matrix(filename, data)
    character(len=*), intent(in) :: filename
    real(r_size), intent(in) :: data(:,:)
    integer(int32) :: nd, dims(2)
    integer :: unit

    nd = 2_int32
    dims = int((/size(data,1), size(data,2)/), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    if (size(data,1) > 0 .and. size(data,2) > 0) write(unit) data
    close(unit)
  END SUBROUTINE write_real_matrix


  SUBROUTINE write_real4d(filename, data)
    character(len=*), intent(in) :: filename
    real(RP), intent(in) :: data(:,:,:,:)
    integer(int32) :: nd, dims(4)
    integer :: unit

    nd = 4_int32
    dims = int((/size(data,1), size(data,2), size(data,3), size(data,4)/), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    write(unit) data
    close(unit)
  END SUBROUTINE write_real4d

  SUBROUTINE write_real3d(filename, data)
    character(len=*), intent(in) :: filename
    real(RP), intent(in) :: data(:,:,:)
    integer(int32) :: nd, dims(3)
    integer :: unit

    nd = 3_int32
    dims = int((/size(data,1), size(data,2), size(data,3)/), int32)

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='replace')
    write(unit) nd
    write(unit) dims
    write(unit) data
    close(unit)
  END SUBROUTINE write_real3d

  SUBROUTINE write_real_vector_prefix(filename, data, count)
    character(len=*), intent(in) :: filename
    real(r_size), intent(in) :: data(:)
    integer, intent(in) :: count
    real(r_size), allocatable :: buffer(:)
    integer :: use_count

    use_count = max(count, 0)
    allocate(buffer(use_count))
    if (use_count > 0) buffer = data(1:use_count)
    call write_real_vector(filename, buffer)
    deallocate(buffer)
  END SUBROUTINE write_real_vector_prefix

  SUBROUTINE write_real_matrix_prefix(filename, data, nrow)
    character(len=*), intent(in) :: filename
    real(r_size), intent(in) :: data(:,:)
    integer, intent(in) :: nrow
    real(r_size), allocatable :: buffer(:,:)
    integer :: ncol
    integer :: use_rows

    ncol = size(data,2)
    use_rows = max(nrow, 0)
    allocate(buffer(use_rows,ncol))
    if (use_rows > 0) buffer = data(1:use_rows,:)
    call write_real_matrix(filename, buffer)
    deallocate(buffer)
  END SUBROUTINE write_real_matrix_prefix

  FUNCTION mem_label(im) RESULT(tag)
    integer, intent(in) :: im
    character(len=memflen) :: tag

    if (im >= 1 .and. im <= MEMBER) then
      write(tag, '(I4.4)') im
    else if (im == mmean) then
      tag = memf_mean
    else if (im == mmdet) then
      tag = memf_mdet
    else if (im == mmgue) then
      tag = memf_mgue
    else
      write(tag, '(I4.4)') im
    end if
  END FUNCTION mem_label

  FUNCTION build_rank_filename(base_dir, prefix, domain_tag, ensemble_tag) RESULT(path)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag
    character(len=filelenmax) :: path

    path = trim(base_dir)//'/'//trim(prefix)//'_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext
  END FUNCTION build_rank_filename

  SUBROUTINE write_state_metadata(dir_local, domain_tag, ensemble_tag)
    character(len=*), intent(in) :: dir_local
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag

    character(len=filelenmax) :: meta_file
    integer :: unit
    integer :: rank_idx
    integer :: tile_i_start_rank, tile_j_start_rank
    integer :: tile_i_size_rank, tile_j_size_rank
    integer :: nij1_rank
    integer :: member_index
    logical :: have_tile
    integer :: tile_ready_flag

    rank_idx = myrank_e + 1
    tile_i_start_rank = -1
    tile_j_start_rank = -1
    tile_i_size_rank = -1
    tile_j_size_rank = -1
    nij1_rank = -1

    member_index = -1
    if (allocated(myrank_to_mem)) then
      if (size(myrank_to_mem) >= 1) then
        member_index = myrank_to_mem(1)
      end if
    end if

    have_tile = tile_partition_ready
    if (have_tile) then
      if (.not. allocated(tile_i_start)) have_tile = .false.
      if (rank_idx < 1) have_tile = .false.
      if (have_tile .and. rank_idx > size(tile_i_start)) have_tile = .false.
    end if

    if (have_tile) then
      tile_i_start_rank = tile_i_start(rank_idx)
      tile_j_start_rank = tile_j_start(rank_idx)
      tile_i_size_rank = tile_i_size(rank_idx)
      tile_j_size_rank = tile_j_size(rank_idx)
    end if

    if (allocated(nij1node)) then
      if (rank_idx >= 1 .and. rank_idx <= size(nij1node)) then
        nij1_rank = nij1node(rank_idx)
      end if
    end if

    meta_file = trim(dir_local)//'/state_meta_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    tile_ready_flag = 0
    if (tile_partition_ready) tile_ready_flag = 1

    open(newunit=unit, file=trim(meta_file), status='replace', action='write')
    write(unit,'(A)') 'meta_version=1'
    write(unit,'(A)') 'domain_suffix='//trim(domain_tag)
    write(unit,'(A)') 'ensemble_suffix='//trim(ensemble_tag)
    write(unit,'(A,I0)') 'myrank=', myrank
    write(unit,'(A,I0)') 'myrank_e=', myrank_e
    write(unit,'(A,I0)') 'myrank_d=', myrank_d
    write(unit,'(A,I0)') 'member_index=', member_index
    write(unit,'(A,I0)') 'rank_index_e=', rank_idx
    write(unit,'(A,I0)') 'nij1=', nij1
    write(unit,'(A,I0)') 'nij1_rank=', nij1_rank
    write(unit,'(A,I0)') 'nlev=', nlev
    write(unit,'(A,I0)') 'nens=', nens
    write(unit,'(A,I0)') 'nv3d=', nv3d
    write(unit,'(A,I0)') 'nv2d=', nv2d
    write(unit,'(A,I0)') 'nlon=', nlon
    write(unit,'(A,I0)') 'nlat=', nlat
    write(unit,'(A,I0)') 'tile_partition_ready=', tile_ready_flag
    write(unit,'(A,I0)') 'tile_i_start=', tile_i_start_rank
    write(unit,'(A,I0)') 'tile_j_start=', tile_j_start_rank
    write(unit,'(A,I0)') 'tile_i_size=', tile_i_size_rank
    write(unit,'(A,I0)') 'tile_j_size=', tile_j_size_rank
    close(unit)
  END SUBROUTINE write_state_metadata

  FUNCTION ensemble_suffix() RESULT(tag)
    character(len=memflen+3) :: tag
    character(len=memflen) :: mem_tag
    integer :: mem_index

    mem_tag = '0000'
    tag = 'mem0000'

    if (allocated(myrank_to_mem)) then
      if (size(myrank_to_mem) >= 1) then
        mem_index = myrank_to_mem(1)
        if (mem_index >= 1) then
          mem_tag = mem_label(mem_index)
        end if
      end if
    end if

    tag = 'mem'//trim(mem_tag)
  END FUNCTION ensemble_suffix

  FUNCTION domain_suffix() RESULT(tag)
    character(len=8) :: tag
    write(tag, '(A2,I6.6)') 'pe', myrank_d
  END FUNCTION domain_suffix

  SUBROUTINE ensure_directory(dir_path)
    character(len=*), intent(in) :: dir_path
    integer :: ierr_local

    if (myrank == 0) then
      call execute_command_line('mkdir -p ' // trim(dir_path), exitstat=ierr_local)
      if (ierr_local /= 0) then
        write(error_unit,'(A,1X,A)') 'letkf_dump: failed to create directory', trim(dir_path)
        stop 1
      end if
    end if
    call MPI_Barrier(MPI_COMM_WORLD, ierr_local)
  END SUBROUTINE ensure_directory

  FUNCTION trim_dir(dir_in) RESULT(dir_out)
    character(len=*), intent(in) :: dir_in
    character(len=filelenmax) :: dir_out
    dir_out = adjustl(dir_in)
    if (len_trim(dir_out) == 0) dir_out = 'letkf_dump'
  END FUNCTION trim_dir

  FUNCTION append_dir(parent, child) RESULT(path)
    character(len=*), intent(in) :: parent
    character(len=*), intent(in) :: child
    character(len=filelenmax) :: path
    path = trim(parent)//'/'//trim(child)
  END FUNCTION append_dir

  FUNCTION parent_dir(dir_in) RESULT(dir_out)
    character(len=*), intent(in) :: dir_in
    character(len=filelenmax) :: dir_out
    integer :: last_sep
    integer :: i
    integer :: end_pos

    dir_out = adjustl(dir_in)
    end_pos = len_trim(dir_out)

    if (end_pos <= 0) then
      dir_out = '.'
      return
    end if

    do while (end_pos > 1)
      if (dir_out(end_pos:end_pos) == '/' .or. dir_out(end_pos:end_pos) == '\') then
        end_pos = end_pos - 1
      else
        exit
      end if
    end do

    last_sep = 0
    do i = 1, end_pos
      if (dir_out(i:i) == '/' .or. dir_out(i:i) == '\') then
        last_sep = i
      end if
    end do

    if (last_sep <= 0) then
      dir_out = '.'
    else if (last_sep == 1 .and. dir_out(1:1) == '/') then
      dir_out = '/'
    else
      dir_out = dir_out(1:last_sep-1)
    end if
  END FUNCTION parent_dir

  SUBROUTINE ensure_obsop_cal_dir()
    character(len=filelenmax) :: dump_parent
    if (obsop_cal_ready) return
    dump_parent = parent_dir(trim_dir(LETKF_INPUT_DUMP_DIR))
    obsop_cal_base = append_dir(dump_parent, 'obsop_cal')
    call ensure_directory(obsop_cal_base)
    obsop_cal_ready = .true.
  END SUBROUTINE ensure_obsop_cal_dir

  FUNCTION obsop_stage_tag(iter, islot) RESULT(tag)
    integer, intent(in) :: iter
    integer, intent(in) :: islot
    character(len=32) :: tag
    write(tag,'(A2,I4.4,A5,I4.4)') 'it', iter, '_slot', islot
  END FUNCTION obsop_stage_tag

  FUNCTION ensemble_suffix_for_mem(mem_index) RESULT(tag)
    integer, intent(in) :: mem_index
    character(len=memflen+3) :: tag
    tag = 'mem'//mem_label(mem_index)
  END FUNCTION ensemble_suffix_for_mem

  SUBROUTINE write_stage_metadata(dir_stage, iter, islot, mem, nobs, n1, n2)
    character(len=*), intent(in) :: dir_stage
    integer, intent(in) :: iter
    integer, intent(in) :: islot
    integer, intent(in) :: mem
    integer, intent(in), optional :: nobs
    integer, intent(in), optional :: n1
    integer, intent(in), optional :: n2
    character(len=filelenmax) :: file_meta
    integer :: unit

    file_meta = append_dir(dir_stage, 'stage_meta.txt')
    open(newunit=unit, file=trim(file_meta), status='replace')
    write(unit,'(A,I0)') 'iter=', iter
    write(unit,'(A,I0)') 'slot=', islot
    write(unit,'(A,I0)') 'member=', mem
    if (present(nobs)) write(unit,'(A,I0)') 'nobs=', nobs
    if (present(n1)) write(unit,'(A,I0)') 'n1=', n1
    if (present(n2)) write(unit,'(A,I0)') 'n2=', n2
    close(unit)
  END SUBROUTINE write_stage_metadata

  SUBROUTINE ensure_directory_local(dir_path)
    character(len=*), intent(in) :: dir_path
    integer :: ierr_local
    logical :: exists
    logical :: am_owner

    inquire(file=trim(dir_path), exist=exists)
    if (exists) return

    am_owner = (das_dump_rank < 0) .or. (myrank == das_dump_rank)
    if (am_owner) then
      call execute_command_line('mkdir -p ' // trim(dir_path), exitstat=ierr_local)
      if (ierr_local /= 0) then
        write(error_unit,'(A,1X,A)') 'letkf_dump: failed to create directory', trim(dir_path)
        stop 1
      end if
    end if
  END SUBROUTINE ensure_directory_local

  SUBROUTINE get_das_phase_dir(stage, phase, dir_phase)
    character(len=*), intent(in) :: stage, phase
    character(len=filelenmax), intent(out) :: dir_phase
    character(len=filelenmax) :: stage_dir

    dir_phase = ''
    if (.not. das_dump_ready()) return

    stage_dir = append_dir(das_base_dir, trim(stage))
    call ensure_directory_local(stage_dir)
    dir_phase = append_dir(stage_dir, trim(phase))
    call ensure_directory_local(dir_phase)
  END SUBROUTINE get_das_phase_dir

  FUNCTION format_call_tag(call_id) RESULT(tag)
    integer(int64), intent(in) :: call_id
    character(len=16) :: tag
    write(tag,'(I12.12)') call_id
  END FUNCTION format_call_tag

  SUBROUTINE write_das_metadata(dir_phase, call_id, lines)
    character(len=*), intent(in) :: dir_phase
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: lines(:)
    character(len=filelenmax) :: meta_file
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    integer :: unit, i
    character(len=16) :: call_tag

    if (.not. das_dump_ready()) return

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    call_tag = format_call_tag(call_id)
    meta_file = trim(dir_phase)//'/meta_call'//trim(call_tag)//'_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    open(newunit=unit, file=trim(meta_file), status='replace', action='write')
    do i=1,size(lines)
      if (len_trim(lines(i)) > 0) then
        write(unit,'(A)') trim(lines(i))
      end if
    end do
    close(unit)
  END SUBROUTINE write_das_metadata

  FUNCTION build_das_rank_filename(base_dir, prefix, call_id, domain_tag, ensemble_tag) RESULT(path)
    character(len=*), intent(in) :: base_dir, prefix
    integer(int64), intent(in) :: call_id
    character(len=*), intent(in) :: domain_tag, ensemble_tag
    character(len=filelenmax) :: path
    character(len=16) :: call_tag

    call_tag = format_call_tag(call_id)
    path = trim(base_dir)//'/'//trim(prefix)//'_call'//trim(call_tag)//'_'//trim(domain_tag)//'.'//trim(ensemble_tag)//dump_suffix_ext
  END FUNCTION build_das_rank_filename

  SUBROUTINE append_meta(meta, count, key, value)
    character(len=*), intent(inout) :: meta(:)
    integer, intent(inout) :: count
    character(len=*), intent(in) :: key, value
    if (count >= size(meta)) return
    count = count + 1
    meta(count) = trim(key)//'='//trim(adjustl(value))
  END SUBROUTINE append_meta

  SUBROUTINE append_meta_int(meta, count, key, value)
    character(len=*), intent(inout) :: meta(:)
    integer, intent(inout) :: count
    character(len=*), intent(in) :: key
    integer(int64), intent(in) :: value
    character(len=64) :: buffer
    write(buffer,'(I0)') value
    call append_meta(meta, count, key, trim(buffer))
  END SUBROUTINE append_meta_int

  SUBROUTINE append_meta_real(meta, count, key, value)
    character(len=*), intent(inout) :: meta(:)
    integer, intent(inout) :: count
    character(len=*), intent(in) :: key
    real(r_size), intent(in) :: value
    character(len=64) :: buffer
    write(buffer,'(ES24.16E3)') value
    call append_meta(meta, count, key, trim(buffer))
  END SUBROUTINE append_meta_real

  SUBROUTINE append_meta_logical(meta, count, key, value)
    character(len=*), intent(inout) :: meta(:)
    integer, intent(inout) :: count
    character(len=*), intent(in) :: key
    logical, intent(in) :: value
    if (value) then
      call append_meta(meta, count, key, 'true')
    else
      call append_meta(meta, count, key, 'false')
    end if
  END SUBROUTINE append_meta_logical

  LOGICAL FUNCTION das_dump_enabled()
    if (.not. LETKF_INPUT_DUMP) then
      das_dump_enabled = .false.
    else if (das_dump_rank < 0) then
      das_dump_enabled = .true.
    else
      das_dump_enabled = (myrank == das_dump_rank)
    end if
  END FUNCTION das_dump_enabled

  LOGICAL FUNCTION das_dump_ready()
    das_dump_ready = das_dump_enabled() .and. das_base_ready
  END FUNCTION das_dump_ready

  LOGICAL FUNCTION das_dump_allow(call_id)
    integer(int64), intent(in) :: call_id
    das_dump_allow = .false.
    if (.not. das_dump_ready()) return
    if (das_dump_max_calls > 0_int64) then
      if (call_id > das_dump_max_calls) return
    end if
    das_dump_allow = .true.
  END FUNCTION das_dump_allow

  SUBROUTINE prepare_das_dump_base()
    character(len=filelenmax) :: root_dir
    character(len=filelenmax) :: dump_root
    character(len=filelenmax) :: dump_parent
    if (.not. das_dump_enabled()) return
    if (das_base_ready) return
    dump_root = trim_dir(LETKF_INPUT_DUMP_DIR)
    dump_parent = parent_dir(dump_root)
    root_dir = append_dir(dump_parent, das_stage_root)
    if (das_dump_rank < 0) then
      call ensure_directory(root_dir)
    else
      call ensure_directory_local(root_dir)
    end if
    das_base_dir = root_dir
    das_base_ready = .true.
    if (.not. das_dump_banner_printed .and. myrank == das_dump_rank) then
      write(6,'(A)') '[das_dump] enabled on this rank; output under '//trim(das_base_dir)
      das_dump_banner_printed = .true.
    end if
  END SUBROUTINE prepare_das_dump_base

  SUBROUTINE das_dump_trace(stage, phase, call_id)
    character(len=*), intent(in) :: stage, phase
    integer(int64), intent(in) :: call_id
    if (das_trace_count >= das_trace_max) return
    das_trace_count = das_trace_count + 1
    write(6,'(A,I0,A,A,A,A,A,A,I0)') '[das_dump] call ', call_id, ' stage=', trim(stage), ' phase=', trim(phase), ' rank=', myrank
  END SUBROUTINE das_dump_trace

END MODULE letkf_dump
