
module letkf_c_api
  use iso_c_binding
  use common
  use common_mpi
  use common_scale
  use common_mpi_scale
  use common_obs_scale
  use common_nml
  use common_mtx, only: mtx_setup
  use letkf_obs
  use letkf_tools
  use obsope_tools, only: obsope_cal
  implicit none
  private

  type(obs_info), allocatable, target, save :: obs_cache(:)
  logical, save :: obs_cache_ready = .false.

  real(r_size), allocatable, target, save :: gues3d_cache(:,:,:,:)
  real(r_size), allocatable, target, save :: gues2d_cache(:,:,:)
  real(r_size), allocatable, target, save :: anal3d_cache(:,:,:,:)
  real(r_size), allocatable, target, save :: anal2d_cache(:,:,:)
  logical, save :: state_cache_ready = .false.

  public :: c_initialize_mpi_scale
  public :: c_finalize_mpi_scale
  public :: c_set_config_path
  public :: c_get_common_nml_info
  public :: c_get_mpi_rank_info
  public :: c_mpi_timer
  public :: c_set_mem_node_proc
  public :: c_set_scalelib
  public :: c_unset_scalelib
  public :: c_set_common_scale
  public :: c_set_common_mpi_scale
  public :: c_unset_common_mpi_scale
  public :: c_set_common_obs_scale
  public :: c_mtx_setup
  public :: c_read_obs_all_mpi
  public :: c_get_obs_in_num
  public :: c_get_obs_nobs
  public :: c_get_obs_component_length
  public :: c_copy_obs_int_component
  public :: c_copy_obs_real_component
  public :: c_obs_info_deallocate
  public :: c_get_nobs_da_mpi
  public :: c_obsope_cal
  public :: c_set_nobs_extern
  public :: c_set_letkf_obs
  public :: c_set_common_mpi_grid
  public :: c_allocate_state_arrays
  public :: c_read_ens_mpi
  public :: c_write_ensmean
  public :: c_write_enssprd
  public :: c_write_anal_sprd
  public :: c_das_letkf
  public :: c_adjust_det_run
  public :: c_ensmean_grd
  public :: c_write_ens_mpi
  public :: c_get_state_dims
  public :: c_copy_state_real

contains

  subroutine c_initialize_mpi_scale(ierr) bind(C, name="letkf_initialize_mpi_scale")
    integer(c_int), intent(out) :: ierr

    call initialize_mpi_scale
    ierr = 0
  end subroutine c_initialize_mpi_scale

  subroutine c_finalize_mpi_scale(ierr) bind(C, name="letkf_finalize_mpi_scale")
    integer(c_int), intent(out) :: ierr

    call finalize_mpi_scale
    ierr = 0
  end subroutine c_finalize_mpi_scale

  subroutine c_set_config_path(path, ierr) bind(C, name="letkf_set_config_path")
    character(kind=c_char), intent(in) :: path(*)
    integer(c_int), intent(out) :: ierr
    character(len=:), allocatable :: fpath

    call c_string_to_fortran(path, fpath)
    call set_common_conf_path(trim(fpath))
    ierr = 0
  end subroutine c_set_config_path

  subroutine c_get_mpi_rank_info(myrank_out, myrank_use_flag, ierr) bind(C, name="letkf_get_mpi_rank_info")
    integer(c_int), intent(out) :: myrank_out
    integer(c_int), intent(out) :: myrank_use_flag
    integer(c_int), intent(out) :: ierr

    myrank_out = myrank
    if (myrank_use) then
      myrank_use_flag = 1
    else
      myrank_use_flag = 0
    end if
    ierr = 0
  end subroutine c_get_mpi_rank_info

  subroutine c_get_common_nml_info(member_out, det_run_out, obsda_in_out, log_level_out, gues_sprd_flag, anal_sprd_flag, departure_stat_flag, ierr) bind(C, name="letkf_get_common_nml_info")
    integer(c_int), intent(out) :: member_out
    integer(c_int), intent(out) :: det_run_out
    integer(c_int), intent(out) :: obsda_in_out
    integer(c_int), intent(out) :: log_level_out
    integer(c_int), intent(out) :: gues_sprd_flag
    integer(c_int), intent(out) :: anal_sprd_flag
    integer(c_int), intent(out) :: departure_stat_flag
    integer(c_int), intent(out) :: ierr

    member_out = MEMBER
    if (DET_RUN) then
      det_run_out = 1
    else
      det_run_out = 0
    end if
    if (OBSDA_IN) then
      obsda_in_out = 1
    else
      obsda_in_out = 0
    end if
    log_level_out = LOG_LEVEL
    if (GUES_SPRD_OUT) then
      gues_sprd_flag = 1
    else
      gues_sprd_flag = 0
    end if
    if (ANAL_SPRD_OUT) then
      anal_sprd_flag = 1
    else
      anal_sprd_flag = 0
    end if
    if (DEPARTURE_STAT) then
      departure_stat_flag = 1
    else
      departure_stat_flag = 0
    end if
    ierr = 0
  end subroutine c_get_common_nml_info

  subroutine c_mpi_timer(name, level, barrier, ierr) bind(C, name="letkf_mpi_timer")
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), value :: level
    integer(c_int), value :: barrier
    integer(c_int), intent(out) :: ierr
    character(len=:), allocatable :: fname

    call c_string_to_fortran(name, fname)
    if (barrier < 0) then
      call mpi_timer(trim(fname), level)
    else
      call mpi_timer(trim(fname), level, barrier)
    end if
    ierr = 0
  end subroutine c_mpi_timer

  subroutine c_set_mem_node_proc(mem, ierr) bind(C, name="letkf_set_mem_node_proc")
    integer(c_int), value :: mem
    integer(c_int), intent(out) :: ierr

    call set_mem_node_proc(mem)
    ierr = 0
  end subroutine c_set_mem_node_proc

  subroutine c_set_scalelib(name, ierr) bind(C, name="letkf_set_scalelib")
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), intent(out) :: ierr
    character(len=:), allocatable :: fname

    call c_string_to_fortran(name, fname)
    call set_scalelib(trim(fname))
    ierr = 0
  end subroutine c_set_scalelib

  subroutine c_unset_scalelib(ierr) bind(C, name="letkf_unset_scalelib")
    integer(c_int), intent(out) :: ierr

    call unset_scalelib
    ierr = 0
  end subroutine c_unset_scalelib

  subroutine c_set_common_scale(ierr) bind(C, name="letkf_set_common_scale")
    integer(c_int), intent(out) :: ierr

    call set_common_scale
    ierr = 0
  end subroutine c_set_common_scale

  subroutine c_set_common_mpi_scale(ierr) bind(C, name="letkf_set_common_mpi_scale")
    integer(c_int), intent(out) :: ierr

    call set_common_mpi_scale
    ierr = 0
  end subroutine c_set_common_mpi_scale

  subroutine c_unset_common_mpi_scale(ierr) bind(C, name="letkf_unset_common_mpi_scale")
    integer(c_int), intent(out) :: ierr

    call unset_common_mpi_scale
    ierr = 0
  end subroutine c_unset_common_mpi_scale

  subroutine c_set_common_obs_scale(ierr) bind(C, name="letkf_set_common_obs_scale")
    integer(c_int), intent(out) :: ierr

    call set_common_obs_scale
    ierr = 0
  end subroutine c_set_common_obs_scale

  subroutine c_mtx_setup(member, ierr) bind(C, name="letkf_mtx_setup")
    integer(c_int), value :: member
    integer(c_int), intent(out) :: ierr

    call mtx_setup(member)
    ierr = 0
  end subroutine c_mtx_setup

  subroutine c_read_obs_all_mpi(ierr) bind(C, name="letkf_read_obs_all_mpi")
    integer(c_int), intent(out) :: ierr

    if (.not. allocated(obs_cache)) allocate(obs_cache(OBS_IN_NUM))
    call read_obs_all_mpi(obs_cache)
    obs_cache_ready = .true.
    ierr = 0
  end subroutine c_read_obs_all_mpi

  subroutine c_get_obs_in_num(count) bind(C, name="letkf_get_obs_in_num")
    integer(c_int), intent(out) :: count

    count = OBS_IN_NUM
  end subroutine c_get_obs_in_num

  subroutine c_get_obs_nobs(iof, nobs, ierr) bind(C, name="letkf_get_obs_nobs")
    integer(c_int), value :: iof
    integer(c_int), intent(out) :: nobs
    integer(c_int), intent(out) :: ierr

    if (.not. obs_cache_ready) then
      ierr = -1
      nobs = 0
      return
    end if

    if (iof < 1 .or. iof > size(obs_cache)) then
      ierr = -2
      nobs = 0
      return
    end if

    nobs = obs_cache(iof)%nobs
    ierr = 0
  end subroutine c_get_obs_nobs

  subroutine c_get_obs_component_length(iof, component, length_out, ierr) bind(C, name="letkf_get_obs_component_length")
    integer(c_int), value :: iof
    integer(c_int), value :: component
    integer(c_int), intent(out) :: length_out
    integer(c_int), intent(out) :: ierr
    integer :: nobs_local

    length_out = 0
    if (.not. obs_cache_ready) then
      ierr = -1
      return
    end if

    if (iof < 1 .or. iof > size(obs_cache)) then
      ierr = -2
      return
    end if

    nobs_local = obs_cache(iof)%nobs

    select case (component)
    case (1,2,3,4,5,6,7,8)
      length_out = nobs_local
    case (9)
      length_out = max_obs_info_meta
    case (10)
      if (allocated(obs_cache(iof)%ri)) length_out = size(obs_cache(iof)%ri)
    case (11)
      if (allocated(obs_cache(iof)%rj)) length_out = size(obs_cache(iof)%rj)
    case (13)
      if (allocated(obs_cache(iof)%rank)) length_out = size(obs_cache(iof)%rank)
    case default
      ierr = -3
      return
    end select

    ierr = 0
  end subroutine c_get_obs_component_length

  subroutine c_copy_obs_int_component(iof, component, dest_ptr, dest_size, ierr) bind(C, name="letkf_copy_obs_int_component")
    integer(c_int), value :: iof
    integer(c_int), value :: component
    type(c_ptr), value :: dest_ptr
    integer(c_int), value :: dest_size
    integer(c_int), intent(out) :: ierr
    integer(c_int), pointer :: dest(:)
    integer :: n

    if (.not. c_associated(dest_ptr)) then
      ierr = -6
      return
    end if

    if (.not. obs_cache_ready) then
      ierr = -1
      return
    end if
    if (iof < 1 .or. iof > size(obs_cache)) then
      ierr = -2
      return
    end if

    select case (component)
    case (1)
      if (.not. allocated(obs_cache(iof)%elm)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      call c_f_pointer(dest_ptr, dest, [dest_size])
      dest(1:n) = obs_cache(iof)%elm
    case (7)
      if (.not. allocated(obs_cache(iof)%typ)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      call c_f_pointer(dest_ptr, dest, [dest_size])
      dest(1:n) = obs_cache(iof)%typ
    case (13)
      if (.not. allocated(obs_cache(iof)%rank)) then
        ierr = -3
        return
      end if
      n = size(obs_cache(iof)%rank)
      if (dest_size < n) then
        ierr = -4
        return
      end if
      call c_f_pointer(dest_ptr, dest, [dest_size])
      dest(1:n) = obs_cache(iof)%rank
    case default
      ierr = -5
      return
    end select

    ierr = 0
  end subroutine c_copy_obs_int_component

  subroutine c_copy_obs_real_component(iof, component, dest_ptr, dest_size, ierr) bind(C, name="letkf_copy_obs_real_component")
    integer(c_int), value :: iof
    integer(c_int), value :: component
    type(c_ptr), value :: dest_ptr
    integer(c_int), value :: dest_size
    integer(c_int), intent(out) :: ierr
    real(c_double), pointer :: dest(:)
    integer :: n

    if (.not. c_associated(dest_ptr)) then
      ierr = -6
      return
    end if

    if (.not. obs_cache_ready) then
      ierr = -1
      return
    end if
    if (iof < 1 .or. iof > size(obs_cache)) then
      ierr = -2
      return
    end if

    call c_f_pointer(dest_ptr, dest, [dest_size])

    select case (component)
    case (2)
      if (.not. allocated(obs_cache(iof)%lon)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%lon
    case (3)
      if (.not. allocated(obs_cache(iof)%lat)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%lat
    case (4)
      if (.not. allocated(obs_cache(iof)%lev)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%lev
    case (5)
      if (.not. allocated(obs_cache(iof)%dat)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%dat
    case (6)
      if (.not. allocated(obs_cache(iof)%err)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%err
    case (8)
      if (.not. allocated(obs_cache(iof)%dif)) then
        ierr = -3
        return
      end if
      n = obs_cache(iof)%nobs
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%dif
    case (9)
      if (dest_size < max_obs_info_meta) then
        ierr = -4
        return
      end if
      dest(1:max_obs_info_meta) = obs_cache(iof)%meta
    case (10)
      if (.not. allocated(obs_cache(iof)%ri)) then
        ierr = -3
        return
      end if
      n = size(obs_cache(iof)%ri)
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%ri
    case (11)
      if (.not. allocated(obs_cache(iof)%rj)) then
        ierr = -3
        return
      end if
      n = size(obs_cache(iof)%rj)
      if (dest_size < n) then
        ierr = -4
        return
      end if
      dest(1:n) = obs_cache(iof)%rj
    case default
      ierr = -5
      return
    end select

    ierr = 0
  end subroutine c_copy_obs_real_component

  subroutine c_obs_info_deallocate(iof, ierr) bind(C, name="letkf_obs_info_deallocate")
    integer(c_int), value :: iof
    integer(c_int), intent(out) :: ierr

    if (.not. allocated(obs_cache)) then
      ierr = -1
      return
    end if
    if (iof < 1 .or. iof > size(obs_cache)) then
      ierr = -2
      return
    end if

    call obs_info_deallocate(obs_cache(iof))
    ierr = 0
  end subroutine c_obs_info_deallocate

  subroutine c_get_nobs_da_mpi(nobs, ierr) bind(C, name="letkf_get_nobs_da_mpi")
    integer(c_int), intent(out) :: nobs
    integer(c_int), intent(out) :: ierr
    integer :: tmp

    call get_nobs_da_mpi(tmp)
    nobs = tmp
    ierr = 0
  end subroutine c_get_nobs_da_mpi

  subroutine c_set_nobs_extern(value, ierr) bind(C, name="letkf_set_nobs_extern")
    integer(c_int), value :: value
    integer(c_int), intent(out) :: ierr

    nobs_extern = value
    ierr = 0
  end subroutine c_set_nobs_extern

  subroutine c_obsope_cal(ierr) bind(C, name="letkf_obsope_cal")
    integer(c_int), intent(out) :: ierr

    call obsope_cal(obsda_return=obsda, nobs_extern=nobs_extern)
    ierr = 0
  end subroutine c_obsope_cal

  subroutine c_set_letkf_obs(ierr) bind(C, name="letkf_set_letkf_obs")
    integer(c_int), intent(out) :: ierr

    call set_letkf_obs
    ierr = 0
  end subroutine c_set_letkf_obs



  subroutine c_set_common_mpi_grid(ierr) bind(C, name="letkf_set_common_mpi_grid")
    integer(c_int), intent(out) :: ierr

    call set_common_mpi_grid
    ierr = 0
  end subroutine c_set_common_mpi_grid

  subroutine c_allocate_state_arrays(ierr) bind(C, name="letkf_allocate_state_arrays")
    integer(c_int), intent(out) :: ierr
    integer :: stat_local

    stat_local = 0
    if (.not. allocated(gues3d_cache)) allocate(gues3d_cache(nij1,nlev,nens,nv3d), stat=stat_local)
    if (stat_local /= 0) then
      ierr = stat_local
      return
    end if

    stat_local = 0
    if (.not. allocated(gues2d_cache)) allocate(gues2d_cache(nij1,nens,nv2d), stat=stat_local)
    if (stat_local /= 0) then
      ierr = stat_local
      return
    end if

    stat_local = 0
    if (.not. allocated(anal3d_cache)) allocate(anal3d_cache(nij1,nlev,nens,nv3d), stat=stat_local)
    if (stat_local /= 0) then
      ierr = stat_local
      return
    end if

    stat_local = 0
    if (.not. allocated(anal2d_cache)) allocate(anal2d_cache(nij1,nens,nv2d), stat=stat_local)
    if (stat_local /= 0) then
      ierr = stat_local
      return
    end if

    state_cache_ready = .true.
    ierr = 0
  end subroutine c_allocate_state_arrays

  subroutine c_read_ens_mpi(ierr) bind(C, name="letkf_read_ens_mpi")
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call read_ens_mpi(gues3d_cache, gues2d_cache)
    ierr = 0
  end subroutine c_read_ens_mpi

  subroutine c_write_ensmean(calced_flag, monit_step, ierr) bind(C, name="letkf_write_ensmean")
    integer(c_int), value :: calced_flag
    integer(c_int), value :: monit_step
    integer(c_int), intent(out) :: ierr
    logical :: calced_f

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    calced_f = (calced_flag /= 0)
    if (monit_step > 0) then
      call write_ensmean(GUES_MEAN_INOUT_BASENAME, gues3d_cache, gues2d_cache, calced=calced_f, monit_step=monit_step)
    else
      call write_ensmean(GUES_MEAN_INOUT_BASENAME, gues3d_cache, gues2d_cache, calced=calced_f)
    end if
    ierr = 0
  end subroutine c_write_ensmean

  subroutine c_write_enssprd(ierr) bind(C, name="letkf_write_enssprd")
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call write_enssprd(GUES_SPRD_OUT_BASENAME, gues3d_cache, gues2d_cache)
    ierr = 0
  end subroutine c_write_enssprd

  subroutine c_write_anal_sprd(ierr) bind(C, name="letkf_write_anal_sprd")
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call write_enssprd(ANAL_SPRD_OUT_BASENAME, anal3d_cache, anal2d_cache)
    ierr = 0
  end subroutine c_write_anal_sprd

  subroutine c_das_letkf(ierr) bind(C, name="letkf_das_letkf")
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call das_letkf(gues3d_cache, gues2d_cache, anal3d_cache, anal2d_cache)
    ierr = 0
  end subroutine c_das_letkf

  subroutine c_adjust_det_run(ierr) bind(C, name="letkf_adjust_det_run")
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    if (DET_RUN) then
      if (mmdetin /= mmdet .and. mmdetin > 0 .and. mmdet > 0) then
        gues3d_cache(:,:,mmdet,:) = gues3d_cache(:,:,mmdetin,:)
        gues2d_cache(:,mmdet,:) = gues2d_cache(:,mmdetin,:)
      end if
    end if

    ierr = 0
  end subroutine c_adjust_det_run


  subroutine c_ensmean_grd(member, nens_local, nij1_local, ierr) bind(C, name="letkf_ensmean_grd")
    integer(c_int), value :: member
    integer(c_int), value :: nens_local
    integer(c_int), value :: nij1_local
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call ensmean_grd(member, nens_local, nij1_local, anal3d_cache, anal2d_cache)
    ierr = 0
  end subroutine c_ensmean_grd

  subroutine c_write_ens_mpi(monit_step, ierr) bind(C, name="letkf_write_ens_mpi")
    integer(c_int), value :: monit_step
    integer(c_int), intent(out) :: ierr

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    if (monit_step > 0) then
      call write_ens_mpi(anal3d_cache, anal2d_cache, monit_step=monit_step)
    else
      call write_ens_mpi(anal3d_cache, anal2d_cache)
    end if
    ierr = 0
  end subroutine c_write_ens_mpi

  subroutine c_get_state_dims(nij1_out, nlev_out, nens_out, nv3d_out, nv2d_out, ierr) bind(C, name="letkf_get_state_dims")
    integer(c_int), intent(out) :: nij1_out
    integer(c_int), intent(out) :: nlev_out
    integer(c_int), intent(out) :: nens_out
    integer(c_int), intent(out) :: nv3d_out
    integer(c_int), intent(out) :: nv2d_out
    integer(c_int), intent(out) :: ierr

    nij1_out = nij1
    nlev_out = nlev
    nens_out = nens
    nv3d_out = nv3d
    nv2d_out = nv2d
    ierr = 0
  end subroutine c_get_state_dims

  subroutine c_copy_state_real(array_id, is_analysis_flag, buffer_ptr, buffer_size, to_fortran_flag, ierr) bind(C, name="letkf_copy_state_real")
    integer(c_int), value :: array_id
    integer(c_int), value :: is_analysis_flag
    type(c_ptr), value :: buffer_ptr
    integer(c_int), value :: buffer_size
    integer(c_int), value :: to_fortran_flag
    integer(c_int), intent(out) :: ierr
    real(c_double), pointer :: buffer(:)
    integer :: required
    logical :: use_analysis
    logical :: copy_to_fortran

    if (.not. c_associated(buffer_ptr)) then
      ierr = -6
      return
    end if

    if (.not. state_cache_ready) then
      ierr = -1
      return
    end if

    call c_f_pointer(buffer_ptr, buffer, [buffer_size])
    use_analysis = (is_analysis_flag /= 0)
    copy_to_fortran = (to_fortran_flag /= 0)

    select case (array_id)
    case (1)
      if (use_analysis) then
        required = size(anal3d_cache)
        if (buffer_size < required) then
          ierr = -3
          return
        end if
        if (copy_to_fortran) then
          anal3d_cache = reshape(buffer(1:required), shape(anal3d_cache))
        else
          buffer(1:required) = reshape(anal3d_cache, [required])
        end if
      else
        required = size(gues3d_cache)
        if (buffer_size < required) then
          ierr = -3
          return
        end if
        if (copy_to_fortran) then
          gues3d_cache = reshape(buffer(1:required), shape(gues3d_cache))
        else
          buffer(1:required) = reshape(gues3d_cache, [required])
        end if
      end if
    case (2)
      if (use_analysis) then
        required = size(anal2d_cache)
        if (buffer_size < required) then
          ierr = -3
          return
        end if
        if (copy_to_fortran) then
          anal2d_cache = reshape(buffer(1:required), shape(anal2d_cache))
        else
          buffer(1:required) = reshape(anal2d_cache, [required])
        end if
      else
        required = size(gues2d_cache)
        if (buffer_size < required) then
          ierr = -3
          return
        end if
        if (copy_to_fortran) then
          gues2d_cache = reshape(buffer(1:required), shape(gues2d_cache))
        else
          buffer(1:required) = reshape(gues2d_cache, [required])
        end if
      end if
    case default
      ierr = -2
      return
    end select

    ierr = 0
  end subroutine c_copy_state_real

  subroutine c_string_to_fortran(cstr, fstr)
    character(kind=c_char), intent(in) :: cstr(*)
    character(len=:), allocatable, intent(out) :: fstr
    integer :: length
    integer :: i

    length = 0
    i = 1
    do
      if (cstr(i) == c_null_char) exit
      length = length + 1
      i = i + 1
    end do

    if (length == 0) then
      allocate(character(len=0) :: fstr)
      return
    end if

    allocate(character(len=length) :: fstr)
    do i = 1, length
      fstr(i:i) = transfer(cstr(i), ' ')
    end do
  end subroutine c_string_to_fortran

end module letkf_c_api
