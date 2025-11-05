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

  SUBROUTINE load_letkf_obs_state(dump_dir)
    character(len=*), intent(in) :: dump_dir

    call import_obsda_dump(trim_dir(dump_dir))
  END SUBROUTINE load_letkf_obs_state

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

    dir_local = base_dir
    call ensure_directory(dir_local)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    call write_state_metadata(dir_local, domain_tag, ensemble_tag)

    if (nv3d > 0) then
      allocate(v3dr(nij1,nlev,nens,nv3d))
      v3dr = real(state3d, RP)
      file3d = build_rank_filename(dir_local, trim(prefix)//'3d', domain_tag, ensemble_tag)
      call write_real4d(file3d, v3dr)
      deallocate(v3dr)
    end if

    if (nv2d > 0) then
      allocate(v2dr(nij1,nens,nv2d))
      v2dr = real(state2d, RP)
      file2d = build_rank_filename(dir_local, trim(prefix)//'2d', domain_tag, ensemble_tag)
      call write_real3d(file2d, v2dr)
      deallocate(v2dr)
    end if

    call MPI_Barrier(MPI_COMM_e, ierr)
  END SUBROUTINE export_state_dump

  SUBROUTINE export_obsda_dump(base_dir)
    character(len=*), intent(in) :: base_dir

    character(len=filelenmax) :: dir_obsda
    character(len=filelenmax) :: meta_file, file_name
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=memflen) :: mem_tag
    integer :: mem, ierr, n_member_dump

    if (.not. allocated(obsda_sort%set)) return

    dir_obsda = trim(base_dir)//'/'//obsda_dir_name
    call ensure_directory(dir_obsda)

    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    meta_file = trim(dir_obsda)//'/obsda_meta_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    call write_obsda_metadata(meta_file)

    file_name = obsda_filename(dir_obsda, memf_mean, domain_tag, ensemble_tag)
    call write_obs_da(file_name, obsda_sort, 0)

    if (allocated(obsda_sort%ensval)) then
      n_member_dump = min(MEMBER, size(obsda_sort%ensval, 1))
    else
      n_member_dump = 0
    end if

    do mem = 1, n_member_dump
      mem_tag = mem_label(mem)
      file_name = obsda_filename(dir_obsda, mem_tag, domain_tag, ensemble_tag)
      call write_obs_da(file_name, obsda_sort, mem)
    end do

    call MPI_Barrier(MPI_COMM_d, ierr)
  END SUBROUTINE export_obsda_dump

  SUBROUTINE import_obsda_dump(base_dir)
    character(len=*), intent(in) :: base_dir

    character(len=filelenmax) :: dir_obsda
    character(len=filelenmax) :: meta_file, file_name
    character(len=8) :: domain_tag
    character(len=memflen+3) :: ensemble_tag
    character(len=memflen) :: mem_tag
    integer :: mem, nobs_local_file, nobs_extern_file

    dir_obsda = trim(base_dir)//'/'//obsda_dir_name
    domain_tag = domain_suffix()
    ensemble_tag = ensemble_suffix()
    meta_file = trim(dir_obsda)//'/obsda_meta_'//trim(domain_tag)//'.'//trim(ensemble_tag)//metadata_ext
    call read_obsda_metadata(meta_file, nobs_local_file, nobs_extern_file)

    obsda%nobs = nobs_local_file
    call obs_da_value_allocate(obsda, MEMBER)
    nobs_extern = nobs_extern_file

    file_name = obsda_filename(dir_obsda, memf_mean, domain_tag, ensemble_tag)
    call read_obs_da(file_name, obsda, 0)

    do mem = 1, MEMBER
      mem_tag = mem_label(mem)
      file_name = obsda_filename(dir_obsda, mem_tag, domain_tag, ensemble_tag)
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

  FUNCTION obsda_filename(dir_obsda, mem_tag, domain_tag, ensemble_tag) RESULT(path)
    character(len=*), intent(in) :: dir_obsda
    character(len=*), intent(in) :: mem_tag
    character(len=*), intent(in) :: domain_tag
    character(len=*), intent(in) :: ensemble_tag
    character(len=filelenmax) :: path

    path = trim(dir_obsda)//'/obsda_'//trim(mem_tag)//'_'//trim(domain_tag)//'.'//trim(ensemble_tag)//'.dat'
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

END MODULE letkf_dump
