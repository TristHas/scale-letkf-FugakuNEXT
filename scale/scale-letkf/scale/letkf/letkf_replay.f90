PROGRAM letkf_replay
  USE common
  USE common_mpi
  USE common_scale
  USE common_mpi_scale
  USE common_obs_scale
  USE common_nml
  use common_mtx, only: mtx_setup
  USE letkf_obs
  USE letkf_tools
  use letkf_dump
  implicit none

  REAL(r_size),ALLOCATABLE :: gues3d(:,:,:,:)
  REAL(r_size),ALLOCATABLE :: gues2d(:,:,:)
  REAL(r_size),ALLOCATABLE :: anal3d(:,:,:,:)
  REAL(r_size),ALLOCATABLE :: anal2d(:,:,:)

  integer :: iof
  character(len=7) :: stdoutf = '-000000'
  character(len=6400) :: icmd
  character(len=filelenmax) :: dump_dir

  call initialize_mpi_scale
  call mpi_timer('', 1)

  if (command_argument_count() >= 2) then
    call get_command_argument(2, icmd)
    if (trim(icmd) /= '') then
      write (stdoutf(2:7), '(I6.6)') myrank
      open (6, file=trim(icmd)//stdoutf)
      write (6,'(A,I6.6,2A)') 'MYRANK=', myrank, ', STDOUTF=', trim(icmd)//stdoutf
    end if
  end if

  if (DET_RUN) then
    call set_mem_node_proc(MEMBER+2)
  else
    call set_mem_node_proc(MEMBER+1)
  end if
  call set_scalelib('LETKF_REPLAY')

  dump_dir = LETKF_INPUT_DUMP_DIR

  if (myrank_use) then

    call set_common_scale
    call set_common_mpi_scale
    call set_common_obs_scale

    call mtx_setup( MEMBER )

    call mpi_timer('INITIALIZE', 1, barrier=MPI_COMM_a)

    allocate (obs(OBS_IN_NUM))
    call read_obs_all_mpi(obs)

    call mpi_timer('READ_OBS', 1, barrier=MPI_COMM_a)

    call load_letkf_obs_state(dump_dir)

    call set_letkf_obs

    call mpi_timer('PROCESS_OBS', 1, barrier=MPI_COMM_a)

    call set_common_mpi_grid

    allocate (gues3d(nij1,nlev,nens,nv3d))
    allocate (gues2d(nij1,nens,nv2d))
    allocate (anal3d(nij1,nlev,nens,nv3d))
    allocate (anal2d(nij1,nens,nv2d))

    call mpi_timer('SET_GRID', 1, barrier=MPI_COMM_a)

    call load_letkf_gues_state(dump_dir, gues3d, gues2d)

    call mpi_timer('READ_GUES', 1, barrier=MPI_COMM_a)

    call das_letkf(gues3d,gues2d,anal3d,anal2d)

    call mpi_timer('DAS_LETKF', 1, barrier=MPI_COMM_a)

    call dump_letkf_analysis_state(anal3d, anal2d, 'anal_replay')

    call mpi_timer('ANAL_DUMP', 1, barrier=MPI_COMM_a)

    do iof = 1, OBS_IN_NUM
      call obs_info_deallocate(obs(iof))
    end do
    deallocate (obs)
    deallocate (gues3d, gues2d, anal3d, anal2d)

    call unset_common_mpi_scale

  end if ! [ myrank_use ]

  call unset_scalelib

  call mpi_timer('FINALIZE', 1, barrier=MPI_COMM_WORLD)

  if ( myrank == 0 ) then
    write(6,'(a)') 'letkf_replay finished successfully'
  endif

  call finalize_mpi_scale

  STOP
END PROGRAM letkf_replay
