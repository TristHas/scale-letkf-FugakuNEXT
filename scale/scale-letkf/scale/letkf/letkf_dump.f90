MODULE letkf_dump
  use iso_fortran_env, only: int32, error_unit
  use scale_precision, only: RP
  use common
  use common_nml
  use common_mpi
  use common_mpi_scale
  use common_scale, only: nlon, nlat, nlev
  use common_obs_scale, only: obs_da_value_allocate, read_obs_da, write_obs_da, obsda_sort
  use letkf_obs, only: obsda, nobs_extern
  implicit none
  private

  character(len=*), parameter :: dump_suffix_ext = '.bin'
  character(len=*), parameter :: obsda_dir_name = 'obsda'
  character(len=*), parameter :: metadata_ext = '.txt'

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

  SUBROUTINE load_letkf_gues_state(dump_dir, gues3d, gues2d)
    character(len=*), intent(in) :: dump_dir
    real(r_size), intent(out) :: gues3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(out) :: gues2d(nij1,nens,nv2d)

    call import_gues_dump(trim_dir(dump_dir), gues3d, gues2d)
  END SUBROUTINE load_letkf_gues_state

  SUBROUTINE load_letkf_obs_state(dump_dir)
    character(len=*), intent(in) :: dump_dir

    call import_obsda_dump(trim_dir(dump_dir))
  END SUBROUTINE load_letkf_obs_state

  SUBROUTINE export_state_dump(base_dir, prefix, state3d, state2d)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    real(r_size), intent(in) :: state3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(in) :: state2d(nij1,nens,nv2d)

    real(RP), allocatable :: v3dg(:,:,:,:)
    real(RP), allocatable :: v2dg(:,:,:)
    integer :: it, im, mstart, mend, ierr
    character(len=filelenmax) :: dir_local

    dir_local = base_dir
    call ensure_directory(dir_local)

    allocate(v3dg(nlev,nlon,nlat,nv3d))
    if (nv2d > 0) then
      allocate(v2dg(nlon,nlat,nv2d))
    else
      allocate(v2dg(1,1,1))
    end if

    do it = 1, nitmax
      im = myrank_to_mem(it)
      mstart = 1 + (it-1)*nprocs_e
      mend   = min(it*nprocs_e, nens)
      if (mstart > mend) cycle

      if ( (im >= 1 .and. im <= MEMBER) ) then
        call gather_grd_mpi_alltoall(mstart, mend, state3d, state2d, v3dg, v2dg)
        call write_member_state(dir_local, prefix, im, v3dg, v2dg)
      else
        call gather_grd_mpi_alltoall(mstart, mend, state3d, state2d, v3dg, v2dg)
      end if
    end do

    deallocate(v3dg)
    deallocate(v2dg)
    call MPI_Barrier(MPI_COMM_e, ierr)
  END SUBROUTINE export_state_dump

  SUBROUTINE write_member_state(dir_local, prefix, im, v3dg, v2dg)
    character(len=*), intent(in) :: dir_local
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: im
    real(RP), intent(in) :: v3dg(nlev,nlon,nlat,nv3d)
    real(RP), intent(in) :: v2dg(:,:,:)

    character(len=filelenmax) :: file3d, file2d

    file3d = build_member_filename(dir_local, trim(prefix)//'3d', im)
    call write_real4d(file3d, v3dg)

    if (nv2d > 0) then
      file2d = build_member_filename(dir_local, trim(prefix)//'2d', im)
      call write_real3d(file2d, v2dg(:,:,1:nv2d))
    end if
  END SUBROUTINE write_member_state

  SUBROUTINE import_gues_dump(base_dir, gues3d, gues2d)
    character(len=*), intent(in) :: base_dir
    real(r_size), intent(out) :: gues3d(nij1,nlev,nens,nv3d)
    real(r_size), intent(out) :: gues2d(nij1,nens,nv2d)

    real(RP), allocatable :: v3dg(:,:,:,:)
    real(RP), allocatable :: v2dg(:,:,:)
    integer :: it, im, mstart, mend
    character(len=filelenmax) :: file3d, file2d

    allocate(v3dg(nlev,nlon,nlat,nv3d))
    if (nv2d > 0) then
      allocate(v2dg(nlon,nlat,nv2d))
    else
      allocate(v2dg(1,1,1))
      v2dg = 0.0_RP
    end if

    do it = 1, nitmax
      im = myrank_to_mem(it)
      mstart = 1 + (it-1)*nprocs_e
      mend   = min(it*nprocs_e, nens)
      if (mstart > mend) cycle

      if (im >= 1 .and. im <= MEMBER) then
        file3d = build_member_filename(base_dir, 'gues3d', im)
        call read_real4d(file3d, v3dg)
        if (nv2d > 0) then
          file2d = build_member_filename(base_dir, 'gues2d', im)
          call read_real3d(file2d, v2dg(:,:,1:nv2d))
        end if
      else
        v3dg = 0.0_RP
        if (nv2d > 0) v2dg(:,:,1:nv2d) = 0.0_RP
      end if

      call scatter_grd_mpi_alltoall(mstart, mend, v3dg, v2dg, gues3d, gues2d)
    end do

    deallocate(v3dg)
    deallocate(v2dg)
  END SUBROUTINE import_gues_dump

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

  SUBROUTINE read_real4d(filename, data)
    character(len=*), intent(in) :: filename
    real(RP), intent(out) :: data(:,:,:,:)
    integer(int32) :: nd, dims(4)
    integer :: unit

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='old')
    read(unit) nd
    if (nd /= 4_int32) then
      write(error_unit, '(A,1X,A)') 'letkf_dump: unexpected rank in file', trim(filename)
      stop 1
    end if
    read(unit) dims
    if (any(dims /= int((/size(data,1), size(data,2), size(data,3), size(data,4)/), int32))) then
      write(error_unit, '(A,1X,A)') 'letkf_dump: size mismatch in file', trim(filename)
      stop 1
    end if
    read(unit) data
    close(unit)
  END SUBROUTINE read_real4d

  SUBROUTINE read_real3d(filename, data)
    character(len=*), intent(in) :: filename
    real(RP), intent(out) :: data(:,:,:)
    integer(int32) :: nd, dims(3)
    integer :: unit

    open(newunit=unit, file=trim(filename), form='unformatted', access='stream', status='old')
    read(unit) nd
    if (nd /= 3_int32) then
      write(error_unit, '(A,1X,A)') 'letkf_dump: unexpected rank in file', trim(filename)
      stop 1
    end if
    read(unit) dims
    if (any(dims /= int((/size(data,1), size(data,2), size(data,3)/), int32))) then
      write(error_unit, '(A,1X,A)') 'letkf_dump: size mismatch in file', trim(filename)
      stop 1
    end if
    read(unit) data
    close(unit)
  END SUBROUTINE read_real3d

  FUNCTION build_member_filename(base_dir, prefix, im) RESULT(path)
    character(len=*), intent(in) :: base_dir
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: im
    character(len=filelenmax) :: path
    character(len=memflen) :: mem_tag

    mem_tag = mem_label(im)
    path = trim(base_dir)//'/'//trim(prefix)//'_'//trim(mem_tag)//dump_suffix_ext
  END FUNCTION build_member_filename

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

END MODULE letkf_dump
