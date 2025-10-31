MODULE letkf_dump
  use iso_fortran_env, only: int32, error_unit, iostat_end
  use scale_precision, only: RP
  use common
  use common_nml
  use common_mpi
  use common_mpi_scale
  use common_scale, only: nlon, nlat, nlev, state_trans, state_trans_inv, &
       write_restart, read_restart
#ifdef PNETCDF
  use common_scale, only: write_restart_par, read_restart_par
  use scale_file, only: FILE_AGGREGATE
#endif
  use common_obs_scale, only: obs_da_value_allocate, read_obs_da, write_obs_da, obsda_sort
  use letkf_obs, only: obsda, nobs_extern
  implicit none
  private

  character(len=*), parameter :: obsda_dir_name = 'obsda'
  character(len=*), parameter :: metadata_ext = '.txt'
  integer, parameter :: copy_chunk_bytes = 1048576

  public :: dump_letkf_obs_state
  public :: dump_letkf_gues_state
  public :: dump_letkf_analysis_state
  public :: load_letkf_gues_state
  public :: load_letkf_obs_state

CONTAINS

  SUBROUTINE dump_letkf_obs_state()
    if (.not. LETKF_INPUT_DUMP) return
    if (.not. allocated(obsda_sort%set)) return
    call export_obsda_dump(trim_dir(LETKF_INPUT_DUMP_DIR))
  END SUBROUTINE dump_letkf_obs_state

  SUBROUTINE dump_letkf_gues_state(gues3d, gues2d)
    real(r_size), intent(in) :: gues3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: gues2d(nij1,nens,nv2d)

    if (.not. LETKF_INPUT_DUMP) return
    call export_state_dump(trim_dir(LETKF_INPUT_DUMP_DIR), 'gues', GUES_IN_BASENAME, gues3d, gues2d)
  END SUBROUTINE dump_letkf_gues_state

  SUBROUTINE dump_letkf_analysis_state(anal3d, anal2d, prefix)
    real(r_size), intent(in) :: anal3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: anal2d(nij1,nens,nv2d)
    character(len=*), intent(in), optional :: prefix
    character(len=32) :: use_prefix
    character(len=filelenmax) :: template_hint

    use_prefix = 'anal'
    if (present(prefix)) use_prefix = prefix

    if (.not. LETKF_INPUT_DUMP) return

    template_hint = ANAL_OUT_BASENAME
    if (len_trim(template_hint) == 0) template_hint = GUES_IN_BASENAME

    call export_state_dump(trim_dir(LETKF_INPUT_DUMP_DIR), use_prefix, template_hint, anal3d, anal2d)
  END SUBROUTINE dump_letkf_analysis_state

  SUBROUTINE load_letkf_gues_state(dump_dir, gues3d, gues2d)
    character(len=*), intent(in) :: dump_dir
    real(r_size), intent(out) :: gues3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(out) :: gues2d(nij1,nens,nv2d)

    call import_state_dump(trim_dir(dump_dir), 'gues', gues3d, gues2d)
  END SUBROUTINE load_letkf_gues_state

  SUBROUTINE load_letkf_obs_state(dump_dir)
    character(len=*), intent(in) :: dump_dir

    call import_obsda_dump(trim_dir(dump_dir))
  END SUBROUTINE load_letkf_obs_state

  SUBROUTINE export_state_dump(base_dir, prefix, template_hint, state3d, state2d)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    character(len=*), intent(in) :: template_hint
    real(r_size), intent(in) :: state3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: state2d(nij1,nens,nv2d)

    real(RP), allocatable :: v3dg(:,:,:,:)
    real(RP), allocatable :: v2dg(:,:,:)
    integer :: it, im, mstart, mend, ierr
    character(len=filelenmax) :: root_dir
    character(len=filelenmax) :: prefix_dir
    character(len=filelenmax) :: member_dir
    character(len=filelenmax) :: target_base
    character(len=filelenmax) :: template_base

    root_dir = base_dir
    call ensure_directory(root_dir)
    prefix_dir = trim(root_dir)//'/'//trim(prefix)
    call ensure_directory(prefix_dir)

    allocate(v3dg(nlev,nlon,nlat,nv3d))
    if (nv2d > 0) then
      allocate(v2dg(nlon,nlat,nv2d))
    else
      allocate(v2dg(nlon,nlat,0))
    end if

    do it = 1, nitmax
      im = myrank_to_mem(it)
      mstart = 1 + (it-1)*nprocs_e
      mend   = min(it*nprocs_e, nens)
      if (mstart > mend) cycle

      call gather_grd_mpi_alltoall(mstart, mend, state3d, state2d, v3dg, v2dg)

      if (im >= 1 .and. im <= MEMBER) then
        call state_trans_inv(v3dg)
        member_dir = build_member_dir(prefix_dir, im)
        call ensure_directory(member_dir)
        target_base = build_state_basename(prefix, member_dir, im)
        template_base = resolve_template_basename(template_hint, im)
        call prepare_state_file(target_base, template_base)
        call write_state_files(target_base, v3dg, v2dg)
      end if
    end do

    deallocate(v3dg)
    if (allocated(v2dg)) deallocate(v2dg)

    call MPI_Barrier(MPI_COMM_e, ierr)
  END SUBROUTINE export_state_dump

  SUBROUTINE import_state_dump(base_dir, prefix, state3d, state2d)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    real(r_size), intent(out) :: state3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(out) :: state2d(nij1,nens,nv2d)

    real(RP), allocatable :: v3dg(:,:,:,:)
    real(RP), allocatable :: v2dg(:,:,:)
    integer :: it, im, mstart, mend
    character(len=filelenmax) :: prefix_dir
    character(len=filelenmax) :: member_dir
    character(len=filelenmax) :: source_base

    prefix_dir = trim(base_dir)//'/'//trim(prefix)

    allocate(v3dg(nlev,nlon,nlat,nv3d))
    if (nv2d > 0) then
      allocate(v2dg(nlon,nlat,nv2d))
    else
      allocate(v2dg(nlon,nlat,0))
    end if

    do it = 1, nitmax
      im = myrank_to_mem(it)
      mstart = 1 + (it-1)*nprocs_e
      mend   = min(it*nprocs_e, nens)
      if (mstart > mend) cycle

      if (im >= 1 .and. im <= MEMBER) then
        member_dir = build_member_dir(prefix_dir, im)
        source_base = build_state_basename(prefix, member_dir, im)
        call read_state_files(source_base, v3dg, v2dg)
        call state_trans(v3dg)
      else
        v3dg = 0.0_RP
        if (nv2d > 0) v2dg = 0.0_RP
      end if

      call scatter_grd_mpi_alltoall(mstart, mend, v3dg, v2dg, state3d, state2d)
    end do

    deallocate(v3dg)
    if (allocated(v2dg)) deallocate(v2dg)
  END SUBROUTINE import_state_dump

  SUBROUTINE export_obsda_dump(base_dir)
    character(len=*), intent(in) :: base_dir

    character(len=filelenmax) :: dir_obsda
    character(len=filelenmax) :: suffix, meta_file, file_name
    character(len=memflen) :: mem_tag
    integer :: mem, ierr, n_member_dump

    if (.not. allocated(obsda_sort%set)) return

    dir_obsda = trim(base_dir)//'/'//obsda_dir_name
    call ensure_directory(dir_obsda)

    suffix = domain_suffix()
    meta_file = trim(dir_obsda)//'/obsda_meta_'//trim(suffix)//metadata_ext
    call write_obsda_metadata(meta_file)

    file_name = obsda_filename(dir_obsda, memf_mean, suffix)
    call write_obs_da(file_name, obsda_sort, 0)

    if (allocated(obsda_sort%ensval)) then
      n_member_dump = min(MEMBER, size(obsda_sort%ensval, 1))
    else
      n_member_dump = 0
    end if

    do mem = 1, n_member_dump
      mem_tag = mem_label(mem)
      file_name = obsda_filename(dir_obsda, mem_tag, suffix)
      call write_obs_da(file_name, obsda_sort, mem)
    end do

    call MPI_Barrier(MPI_COMM_d, ierr)
  END SUBROUTINE export_obsda_dump

  SUBROUTINE import_obsda_dump(base_dir)
    character(len=*), intent(in) :: base_dir

    character(len=filelenmax) :: dir_obsda
    character(len=filelenmax) :: suffix, meta_file, file_name
    character(len=memflen) :: mem_tag
    integer :: mem, nobs_local_file, nobs_extern_file

    dir_obsda = trim(base_dir)//'/'//obsda_dir_name
    suffix = domain_suffix()
    meta_file = trim(dir_obsda)//'/obsda_meta_'//trim(suffix)//metadata_ext
    call read_obsda_metadata(meta_file, nobs_local_file, nobs_extern_file)

    obsda%nobs = nobs_local_file
    call obs_da_value_allocate(obsda, MEMBER)
    nobs_extern = nobs_extern_file

    file_name = obsda_filename(dir_obsda, memf_mean, suffix)
    call read_obs_da(file_name, obsda, 0)

    do mem = 1, MEMBER
      mem_tag = mem_label(mem)
      file_name = obsda_filename(dir_obsda, mem_tag, suffix)
      call read_obs_da(file_name, obsda, mem)
    end do
  END SUBROUTINE import_obsda_dump

  SUBROUTINE write_obsda_metadata(filename)
    character(len=*), intent(in) :: filename
    integer :: unit

    open(newunit=unit, file=trim(filename), status='replace', action='write')
    write(unit,'(A,I0)') 'nobs=', obsda_sort%nobs
    write(unit,'(A,I0)') 'nobs_extern=', nobs_extern
    write(unit,'(A,I0)') 'member_count=', MEMBER
    close(unit)
  END SUBROUTINE write_obsda_metadata

  SUBROUTINE read_obsda_metadata(filename, nobs_local_file, nobs_extern_file)
    character(len=*), intent(in) :: filename
    integer, intent(out) :: nobs_local_file
    integer, intent(out) :: nobs_extern_file

    character(len=256) :: line
    integer :: unit, eqpos, ios

    nobs_local_file = 0
    nobs_extern_file = 0

    open(newunit=unit, file=trim(filename), status='old', action='read')
    do
      read(unit,'(A)', iostat=ios) line
      if (ios /= 0) exit
      eqpos = index(line, '=')
      if (eqpos <= 0) cycle
      select case (adjustl(line(1:eqpos-1)))
      case ('nobs')
        read(line(eqpos+1:),*) nobs_local_file
      case ('nobs_extern')
        read(line(eqpos+1:),*) nobs_extern_file
      end select
    end do
    close(unit)
  END SUBROUTINE read_obsda_metadata

  FUNCTION obsda_filename(dir_obsda, mem_tag, suffix) RESULT(path)
    character(len=*), intent(in) :: dir_obsda
    character(len=*), intent(in) :: mem_tag
    character(len=*), intent(in) :: suffix
    character(len=filelenmax) :: path

    path = trim(dir_obsda)//'/obsda_'//trim(mem_tag)//'_'//trim(suffix)//'.dat'
  END FUNCTION obsda_filename

  SUBROUTINE ensure_directory(dir_path)
    character(len=*), intent(in) :: dir_path
    integer :: ierr_local

    if (len_trim(dir_path) == 0) return

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

  FUNCTION domain_suffix() RESULT(tag)
    character(len=8) :: tag
    write(tag, '(A2,I6.6)') 'pe', myrank_d
  END FUNCTION domain_suffix

  FUNCTION build_member_dir(prefix_dir, im) RESULT(path)
    character(len=*), intent(in) :: prefix_dir
    integer, intent(in) :: im
    character(len=filelenmax) :: path

    path = trim(prefix_dir)//'/'//trim(mem_label(im))
  END FUNCTION build_member_dir

  FUNCTION build_state_basename(prefix, member_dir, im) RESULT(path)
    character(len=*), intent(in) :: prefix
    character(len=*), intent(in) :: member_dir
    integer, intent(in) :: im
    character(len=filelenmax) :: path

    path = trim(member_dir)//'/'//trim(prefix)//'.'//trim(mem_label(im))
  END FUNCTION build_state_basename

  FUNCTION resolve_template_basename(hint, im) RESULT(path)
    character(len=*), intent(in) :: hint
    integer, intent(in) :: im
    character(len=filelenmax) :: path

    path = hint
    if (len_trim(path) == 0) path = GUES_IN_BASENAME
    call filename_replace_mem(path, im)
    if (len_trim(path) == 0) then
      write(error_unit,'(A,I0)') 'letkf_dump: unable to determine template basename for member', im
      stop 1
    end if
  END FUNCTION resolve_template_basename

  FUNCTION state_file_path(filename_base) RESULT(path)
    character(len=*), intent(in) :: filename_base
    character(len=filelenmax) :: path

#ifdef PNETCDF
    if (FILE_AGGREGATE) then
      path = trim(filename_base)//'.nc'
    else
#endif
      path = trim(filename_base)//'.'//trim(domain_suffix())//'.nc'
#ifdef PNETCDF
    end if
#endif
  END FUNCTION state_file_path

  SUBROUTINE prepare_state_file(target_base, template_base)
    character(len=*), intent(in) :: target_base
    character(len=*), intent(in) :: template_base
    character(len=filelenmax) :: target_file
    character(len=filelenmax) :: template_file
    logical :: exists_target, exists_template
    integer :: ierr_local

    target_file = state_file_path(target_base)
    inquire(file=trim(target_file), exist=exists_target)
    if (exists_target) return

    template_file = state_file_path(template_base)
    inquire(file=trim(template_file), exist=exists_template)
    if (.not. exists_template) then
      write(error_unit,'(A,1X,A)') 'letkf_dump: template file missing', trim(template_file)
      stop 1
    end if

#ifdef PNETCDF
    if (FILE_AGGREGATE) then
      if (myrank_d == 0) call copy_file_stream(template_file, target_file)
      call MPI_Barrier(MPI_COMM_d, ierr_local)
    else
      call copy_file_stream(template_file, target_file)
    end if
#else
    call copy_file_stream(template_file, target_file)
#endif
  END SUBROUTINE prepare_state_file

  SUBROUTINE write_state_files(filename_base, v3dg, v2dg)
    character(len=*), intent(in) :: filename_base
    real(RP), intent(inout) :: v3dg(nlev,nlon,nlat,nv3d)
    real(RP), intent(in) :: v2dg(nlon,nlat,*)
#ifdef PNETCDF
    if (FILE_AGGREGATE) then
      call write_restart_par(filename_base, v3dg, v2dg, MPI_COMM_d)
    else
#endif
      call write_restart(filename_base, v3dg, v2dg)
#ifdef PNETCDF
    end if
#endif
  END SUBROUTINE write_state_files

  SUBROUTINE read_state_files(filename_base, v3dg, v2dg)
    character(len=*), intent(in) :: filename_base
    real(RP), intent(out) :: v3dg(nlev,nlon,nlat,nv3d)
    real(RP), intent(inout) :: v2dg(nlon,nlat,*)
#ifdef PNETCDF
    if (FILE_AGGREGATE) then
      call read_restart_par(filename_base, v3dg, v2dg, MPI_COMM_d)
    else
#endif
      call read_restart(filename_base, v3dg, v2dg)
#ifdef PNETCDF
    end if
#endif
  END SUBROUTINE read_state_files

  SUBROUTINE copy_file_stream(src, dest)
    character(len=*), intent(in) :: src
    character(len=*), intent(in) :: dest
    integer :: in_unit, out_unit, ios, ios_write, read_size
    character(len=copy_chunk_bytes) :: buffer

    open(newunit=in_unit, file=trim(src), status='old', action='read', access='stream', form='unformatted')
    open(newunit=out_unit, file=trim(dest), status='replace', action='write', access='stream', form='unformatted')

    do
      read(in_unit, iostat=ios, size=read_size) buffer
      if (ios == iostat_end) then
        if (read_size > 0) then
          write(out_unit, iostat=ios_write, size=read_size) buffer
          if (ios_write /= 0) then
            write(error_unit,'(A,1X,A)') 'letkf_dump: failed to copy file (write)', trim(dest)
            stop 1
          end if
        end if
        exit
      else if (ios /= 0) then
        write(error_unit,'(A,1X,A)') 'letkf_dump: failed to copy file (read)', trim(src)
        stop 1
      end if

      write(out_unit, iostat=ios_write, size=read_size) buffer
      if (ios_write /= 0) then
        write(error_unit,'(A,1X,A)') 'letkf_dump: failed to copy file (write)', trim(dest)
        stop 1
      end if
    end do

    close(out_unit)
    close(in_unit)
  END SUBROUTINE copy_file_stream

END MODULE letkf_dump
