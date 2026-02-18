MODULE obsope_tools
!=======================================================================
!
! [PURPOSE:] Observation operator tools
!
! [HISTORY:]
!   November 2014  Guo-Yuan Lien  created
!   .............  See git history for the following revisions
!
!=======================================================================
!$USE OMP_LIB
  use iso_fortran_env, only: int32
  USE common
  USE common_mpi
  USE common_scale
  USE common_mpi_scale
  USE common_obs_scale
  use common_nml
!  use scale_prc, only: &
!    PRC_myrank
!    MPI_COMM_d => LOCAL_COMM_WORLD
  use scale_atmos_grid_cartesC_index, only: &
    KHALO, IHALO, JHALO

  IMPLICIT NONE
  PUBLIC

  character(len=filelenmax), parameter :: obsop_dump_dirname = 'obsop_cal'
  character(len=*), parameter :: obsop_dump_suffix = '.bin'
  integer, parameter :: obsop_dump_target_rank = 0  ! set to -1 to dump all ranks
  character(len=filelenmax), save :: obsop_base_dir = ''
  logical, save :: obsop_dir_ready = .false.

CONTAINS

!-----------------------------------------------------------------------
! Observation operator calculation
!-----------------------------------------------------------------------
SUBROUTINE obsope_cal(obsda_return, nobs_extern)
  IMPLICIT NONE

  type(obs_da_value), optional, intent(out) :: obsda_return
  integer, optional, intent(in) :: nobs_extern

  type(obs_da_value) :: obsda

  integer :: it, im, iof, islot, ierr
  integer :: n, nn, nsub, nmod, n1, n2
  integer :: nobsl, nobsl_dump, islot2, iloc

  integer :: nobs     ! observation number processed in this subroutine
  integer :: nobs_all
  integer :: nobs_max_per_file
  integer :: nobs_max_per_file_sub
  integer :: slot_nobsg

  integer :: ip, ibufs
  integer, allocatable :: cntr(:), dspr(:)
  integer, allocatable :: cnts(:), dsps(:)
  integer, allocatable :: bsn(:,:), bsna(:,:), bsnext(:,:)
  integer :: islot_time_out, islot_domain_out

  integer, allocatable :: obrank_bufs(:)
  real(r_size), allocatable :: ri_bufs(:)
  real(r_size), allocatable :: rj_bufs(:)

  integer, allocatable :: obset_bufs(:)
  integer, allocatable :: obidx_bufs(:)

  integer :: slot_id(SLOT_START:SLOT_END)
  real(r_size) :: slot_lb(SLOT_START:SLOT_END)
  real(r_size) :: slot_ub(SLOT_START:SLOT_END)

  real(r_size), allocatable :: v3dg(:,:,:,:)
  real(r_size), allocatable :: v2dg(:,:,:)

  real(r_size), allocatable :: mv3dg(:,:,:,:,:)
  real(r_size), allocatable :: mv2dg(:,:,:,:)

  real(r_size), allocatable :: mdbz3dg(:,:,:,:)

  type(obsop_interp_dump_context) :: dump_ctx
  real(r_size), allocatable :: slope3dg(:,:,:)

  real(r_size) :: ril, rjl, rk, rkz

  integer, allocatable :: dump_sets(:)
  integer, allocatable :: dump_idx(:)
  integer, allocatable :: dump_nn(:)
  integer, allocatable :: dump_elms(:)
  integer, allocatable :: dump_typs(:)
  integer, allocatable :: dump_qc(:)
  real(r_size), allocatable :: dump_lon(:)
  real(r_size), allocatable :: dump_lat(:)
  real(r_size), allocatable :: dump_lev(:)
  real(r_size), allocatable :: dump_meta(:,:)
  real(r_size), allocatable :: dump_ri_global(:)
  real(r_size), allocatable :: dump_rj_global(:)
  real(r_size), allocatable :: dump_ril(:)
  real(r_size), allocatable :: dump_rjl(:)
  real(r_size), allocatable :: dump_rkz(:)

  character(filelenmax) :: obsdafile
  character(len=11) :: obsda_suffix = '.000000.dat'
  character(len=4) :: nstr
  character(len=timer_name_width) :: timer_str

!-------------------------------------------------------------------------------

  call mpi_timer('', 2)

!-------------------------------------------------------------------------------
! First scan of all observation data: Compute their horizontal location and time
!-------------------------------------------------------------------------------

  nobs_all = 0
  nobs_max_per_file = 0
  do iof = 1, OBS_IN_NUM
    if (obs(iof)%nobs > nobs_max_per_file) then
      nobs_max_per_file = obs(iof)%nobs
    end if
    if (OBSDA_RUN(iof)) then
      nobs_all = nobs_all + obs(iof)%nobs
    end if
  end do

!  obs_set_TCX = -1
!  obs_set_TCY = -1
!  obs_set_TCP = -1
!  obs_idx_TCX = -1
!  obs_idx_TCY = -1
!  obs_idx_TCP = -1

  nobs_max_per_file_sub = (nobs_max_per_file - 1) / nprocs_a + 1
  allocate (obrank_bufs(nobs_max_per_file_sub))
  allocate (ri_bufs(nobs_max_per_file_sub))
  allocate (rj_bufs(nobs_max_per_file_sub))

  allocate (cntr(nprocs_a))
  allocate (dspr(nprocs_a))

  ! Use all processes to compute the basic obsevration information
  ! (locations in model grids and the subdomains they belong to)
  !-----------------------------------------------------------------------------

  do iof = 1, OBS_IN_NUM
    if (obs(iof)%nobs > 0) then ! Process basic obsevration information for all observations since this information is not saved in obsda files
                                ! when using separate observation operators; ignore the 'OBSDA_RUN' setting for this section
      nsub = obs(iof)%nobs / nprocs_a
      nmod = mod(obs(iof)%nobs, nprocs_a)
      do ip = 1, nmod
        cntr(ip) = nsub + 1
      end do
      do ip = nmod+1, nprocs_a
        cntr(ip) = nsub
      end do
      dspr(1) = 0
      do ip = 2, nprocs_a
        dspr(ip) = dspr(ip-1) + cntr(ip-1)
      end do

      obrank_bufs(:) = -1
!$OMP PARALLEL DO PRIVATE(ibufs,n) SCHEDULE(STATIC)
      do ibufs = 1, cntr(myrank_a+1)
        n = dspr(myrank_a+1) + ibufs
!        select case (obs(iof)%elm(n))
!        case (id_tclon_obs)
!          obs_set_TCX = iof
!          obs_idx_TCX = n
!          cycle
!        case (id_tclat_obs)
!          obs_set_TCY = iof
!          obs_idx_TCY = n
!          cycle
!        case (id_tcmip_obs)
!          obs_set_TCP = iof
!          obs_idx_TCP = n
!          cycle
!        end select

        call phys2ij(obs(iof)%lon(n), obs(iof)%lat(n), ri_bufs(ibufs), rj_bufs(ibufs))
        call rij_rank(ri_bufs(ibufs), rj_bufs(ibufs), obrank_bufs(ibufs))
      end do ! [ ibufs = 1, cntr(myrank_a+1) ]
!$OMP END PARALLEL DO

      call mpi_timer('obsope_cal:first_scan_cal:', 2, barrier=MPI_COMM_a)

      call MPI_ALLGATHERV(obrank_bufs, cntr(myrank_a+1), MPI_INTEGER, obs(iof)%rank, cntr, dspr, MPI_INTEGER, MPI_COMM_a, ierr)
      call MPI_ALLGATHERV(ri_bufs,     cntr(myrank_a+1), MPI_r_size,  obs(iof)%ri,   cntr, dspr, MPI_r_size,  MPI_COMM_a, ierr)
      call MPI_ALLGATHERV(rj_bufs,     cntr(myrank_a+1), MPI_r_size,  obs(iof)%rj,   cntr, dspr, MPI_r_size,  MPI_COMM_a, ierr)

      call mpi_timer('obsope_cal:first_scan_reduce:', 2)
    end if ! [ obs(iof)%nobs > 0 ]
  end do ! [ do iof = 1, OBS_IN_NUM ]

  deallocate (cntr, dspr)
  deallocate (obrank_bufs, ri_bufs, rj_bufs)

  ! Bucket sort of observation wrt. time slots and subdomains using the process rank 0
  !-----------------------------------------------------------------------------

  islot_time_out = SLOT_END + 1   ! slot = SLOT_END+1 for observation not in the assimilation time window
  islot_domain_out = SLOT_END + 2 ! slot = SLOT_END+2 for observation outside of the model domain

  allocate (bsn   (SLOT_START  :SLOT_END+2, 0:nprocs_d-1))
  allocate (bsna  (SLOT_START-1:SLOT_END+2, 0:nprocs_d-1))

  if (myrank_e == 0) then
    allocate ( obset_bufs(nobs_all) )
    allocate ( obidx_bufs(nobs_all) )
  end if

  if (myrank_a == 0) then
    allocate (bsnext(SLOT_START  :SLOT_END+2, 0:nprocs_d-1))
    bsn(:,:) = 0
    bsna(:,:) = 0
    bsnext(:,:) = 0

!$OMP PARALLEL PRIVATE(iof,n,islot)
    do iof = 1, OBS_IN_NUM
      if (OBSDA_RUN(iof) .and. obs(iof)%nobs > 0) then
!$OMP DO SCHEDULE(STATIC)
        do n = 1, obs(iof)%nobs
          if (obs(iof)%rank(n) == -1) then
            ! process the observations outside of the model domain in process rank 0
!$OMP ATOMIC
            bsn(islot_domain_out, 0) = bsn(islot_domain_out, 0) + 1
          else
            islot = ceiling(obs(iof)%dif(n) / SLOT_TINTERVAL - 0.5d0) + SLOT_BASE
            if (islot < SLOT_START .or. islot > SLOT_END) then
              islot = islot_time_out
            end if
!$OMP ATOMIC
            bsn(islot, obs(iof)%rank(n)) = bsn(islot, obs(iof)%rank(n)) + 1
          end if
        end do ! [ n = 1, obs(iof)%nobs ]
!$OMP END DO
      end if ! [ OBSDA_RUN(iof) .and. obs(iof)%nobs > 0 ]
    end do ! [ do iof = 1, OBS_IN_NUM ]
!$OMP END PARALLEL

    do ip = 0, nprocs_d-1
      if (ip > 0) then
        bsna(SLOT_START-1, ip) = bsna(SLOT_END+2, ip-1)
      end if
      do islot = SLOT_START, SLOT_END+2
        bsna(islot, ip) = bsna(islot-1, ip) + bsn(islot, ip)
      end do
      bsnext(SLOT_START:SLOT_END+2, ip) = bsna(SLOT_START-1:SLOT_END+1, ip)
    end do

    do iof = 1, OBS_IN_NUM
      if (OBSDA_RUN(iof) .and. obs(iof)%nobs > 0) then
        do n = 1, obs(iof)%nobs
          if (obs(iof)%rank(n) == -1) then
            ! process the observations outside of the model domain in process rank 0
            bsnext(islot_domain_out, 0) = bsnext(islot_domain_out, 0) + 1
            obset_bufs(bsnext(islot_domain_out, 0)) = iof
            obidx_bufs(bsnext(islot_domain_out, 0)) = n
          else
            islot = ceiling(obs(iof)%dif(n) / SLOT_TINTERVAL - 0.5d0) + SLOT_BASE
            if (islot < SLOT_START .or. islot > SLOT_END) then
              islot = islot_time_out
            end if
            bsnext(islot, obs(iof)%rank(n)) = bsnext(islot, obs(iof)%rank(n)) + 1
            obset_bufs(bsnext(islot, obs(iof)%rank(n))) = iof
            obidx_bufs(bsnext(islot, obs(iof)%rank(n))) = n
          end if
        end do ! [ n = 1, obs(iof)%nobs ]
      end if ! [ OBSDA_RUN(iof) .and. obs(iof)%nobs > 0 ]
    end do ! [ do iof = 1, OBS_IN_NUM ]

    deallocate (bsnext)

    call mpi_timer('obsope_cal:bucket_sort:', 2)
  end if ! [ myrank_a == 0 ]

  ! Broadcast the bucket-sort observation numbers to all processes and print
  !-----------------------------------------------------------------------------

  call mpi_timer('', 2, barrier=MPI_COMM_a)

  call MPI_BCAST(bsn,  (SLOT_END-SLOT_START+3)*nprocs_d, MPI_INTEGER, 0, MPI_COMM_a, ierr)
  call MPI_BCAST(bsna, (SLOT_END-SLOT_START+4)*nprocs_d, MPI_INTEGER, 0, MPI_COMM_a, ierr)

  call mpi_timer('obsope_cal:sort_info_bcast:', 2)

  do islot = SLOT_START, SLOT_END
    slot_id(islot) = islot - SLOT_START + 1
    slot_lb(islot) = (real(islot - SLOT_BASE, r_size) - 0.5d0) * SLOT_TINTERVAL
    slot_ub(islot) = (real(islot - SLOT_BASE, r_size) + 0.5d0) * SLOT_TINTERVAL
  end do

  if (LOG_LEVEL >= 2) then
    write (nstr, '(I4)') SLOT_END - SLOT_START + 1
    write (6, *)
    write (6, '(A,I6,A)') 'OBSERVATION COUNTS BEFORE QC (FROM OBSOPE):'
    write (6, '(A,'//nstr//"('=========='),A)") '====================', '===================='
    write (6, '(A,'//nstr//'I10.4)') '            SLOT #  ', slot_id(:)
    write (6, '(A,'//nstr//'F10.1)') '            FROM (s)', slot_lb(:)
    write (6, '(A,'//nstr//'F10.1,A)') 'SUBDOMAIN #   TO (s)', slot_ub(:), '  OUT_TIME     TOTAL'
    write (6, '(A,'//nstr//"('----------'),A)") '--------------------', '--------------------'
    do ip = 0, nprocs_d-1
      write (6, '(I11.6,9x,'//nstr//'I10,2I10)') ip, bsn(SLOT_START:SLOT_END, ip), bsn(islot_time_out, ip), bsna(SLOT_END+1, ip) - bsna(SLOT_START-1, ip)
    end do
    write (6, '(A,'//nstr//"('----------'),A)") '--------------------', '--------------------'
    write (6, '(A,'//nstr//'(10x),10x,I10)') ' OUT_DOMAIN         ', bsn(islot_domain_out, 0)
    write (6, '(A,'//nstr//"('----------'),A)") '--------------------', '--------------------'
    write (6, '(A,'//nstr//'I10,2I10)') '      TOTAL         ', sum(bsn(SLOT_START:SLOT_END, :), dim=2), sum(bsn(islot_time_out, :)), bsna(SLOT_END+2, nprocs_d-1)
    write (6, '(A,'//nstr//"('=========='),A)") '====================', '===================='
  end if

  ! Scatter the basic obsevration information to processes group {myrank_e = 0},
  ! each of which only gets the data in its own subdomain
  !-----------------------------------------------------------------------------

  nobs = bsna(SLOT_END+2, myrank_d) - bsna(SLOT_START-1, myrank_d)

  obsda%nobs = nobs
  call obs_da_value_allocate(obsda, 0)

  if (present(obsda_return)) then
    if (present(nobs_extern)) then
      obsda_return%nobs = nobs + nobs_extern
    else
      obsda_return%nobs = nobs
    end if
    call obs_da_value_allocate(obsda_return, nitmax)
  end if

  if (myrank_e == 0) then
    allocate (cnts(nprocs_d))
    allocate (dsps(nprocs_d))
    do ip = 0, nprocs_d-1
      dsps(ip+1) = bsna(SLOT_START-1, ip)
      cnts(ip+1) = bsna(SLOT_END+2, ip) - dsps(ip+1)
    end do

    call MPI_SCATTERV(obset_bufs, cnts, dsps, MPI_INTEGER, obsda%set, cnts(myrank_d+1), MPI_INTEGER, 0, MPI_COMM_d, ierr)
    call MPI_SCATTERV(obidx_bufs, cnts, dsps, MPI_INTEGER, obsda%idx, cnts(myrank_d+1), MPI_INTEGER, 0, MPI_COMM_d, ierr)

    call mpi_timer('obsope_cal:mpi_scatterv:', 2)

    deallocate (cnts, dsps)
    deallocate (obset_bufs, obidx_bufs)
  end if ! [ myrank_e == 0 ]

  ! Broadcast the basic obsevration information
  ! from processes group {myrank_e = 0} to all processes
  !-----------------------------------------------------------------------------

  call mpi_timer('', 2, barrier=MPI_COMM_e)

  call MPI_BCAST(obsda%set, nobs, MPI_INTEGER, 0, MPI_COMM_e, ierr)
  call MPI_BCAST(obsda%idx, nobs, MPI_INTEGER, 0, MPI_COMM_e, ierr)

  if (present(obsda_return)) then
    obsda_return%set(1:nobs) = obsda%set
    obsda_return%idx(1:nobs) = obsda%idx
  end if

  call mpi_timer('obsope_cal:mpi_broadcast:', 2)

!-------------------------------------------------------------------------------
! Second scan of observation data in own subdomain: Compute H(x), QC, ... etc.
!-------------------------------------------------------------------------------

  allocate ( v3dg (nlevh,nlonh,nlath,nv3dd) )
  allocate ( v2dg (nlonh,nlath,nv2dd) )

  if ( RADAR_ADDITIVE_Y18 ) then
    allocate ( mv3dg (SLOT_END-SLOT_START+1,nlevh,nlonh,nlath,nv3dd) )
    allocate ( mv2dg (SLOT_END-SLOT_START+1,nlonh,nlath,nv2dd) )
    allocate ( mdbz3dg(SLOT_END-SLOT_START+1,nlev,nlon,nlat) )

    allocate ( slope3dg (SLOT_END-SLOT_START+1,nlevh,nv3dd) )

    call get_history_ensemble_mean_mpi( mv3dg, mv2dg, mdbz3dg )
    call mpi_timer('obsope_cal:get_mean_Y18:', 2)
    call get_regression_slope_dbz_mpi( mv3dg, mv2dg, mdbz3dg, slope3dg )
    call mpi_timer('obsope_cal:get_slope_Y18:', 2)
  endif

  do it = 1, nitmax
    im = myrank_to_mem(it)
    if ((im >= 1 .and. im <= MEMBER) .or. im == mmdetin) then

      if ( LOG_OUT ) write (6,'(A,I6.6,A,I4.4,A,I6.6)') 'MYRANK ',myrank,' is processing member ', &
                                                        im, ', subdomain id #', myrank_d

      if (nobs > 0) then
        obsda%qc(1:nobs) = iqc_undef
      end if

      ! Observations not in the assimilation time window
      ! 
      n1 = bsna(islot_time_out-1, myrank_d) - bsna(SLOT_START-1, myrank_d) + 1
      n2 = bsna(islot_time_out,   myrank_d) - bsna(SLOT_START-1, myrank_d)
      if (n1 <= n2) then
        obsda%qc(n1:n2) = iqc_time
      end if

      ! Observations outside of the model domain
      ! 
      n1 = bsna(islot_domain_out-1, myrank_d) - bsna(SLOT_START-1, myrank_d) + 1
      n2 = bsna(islot_domain_out,   myrank_d) - bsna(SLOT_START-1, myrank_d)
      if (n1 <= n2) then
        obsda%qc(n1:n2) = iqc_out_h
      end if

      ! Valid observations: loop over time slots
      ! 
      do islot = SLOT_START, SLOT_END
        if ( LOG_OUT )write (6, '(A,I3,A,F9.1,A,F9.1,A)') 'Slot #', islot-SLOT_START+1, ': time window (', slot_lb(islot), ',', slot_ub(islot), '] sec'

        n1 = bsna(islot-1, myrank_d) - bsna(SLOT_START-1, myrank_d) + 1
        n2 = bsna(islot,   myrank_d) - bsna(SLOT_START-1, myrank_d)
        slot_nobsg = sum(bsn(islot, :))
        islot2 = islot - SLOT_START + 1
        nobsl = max(n2 - n1 + 1, 0)
        if (obsop_dump_active()) then
          nobsl_dump = nobsl
        else
          nobsl_dump = 0
        end if
        if (nobsl_dump > 0) then
          allocate(dump_sets(nobsl))
          allocate(dump_idx(nobsl))
          allocate(dump_nn(nobsl))
          allocate(dump_elms(nobsl))
          allocate(dump_typs(nobsl))
          allocate(dump_qc(nobsl))
          allocate(dump_lon(nobsl))
          allocate(dump_lat(nobsl))
          allocate(dump_lev(nobsl))
          allocate(dump_meta(max_obs_info_meta,nobsl))
          allocate(dump_ri_global(nobsl))
          allocate(dump_rj_global(nobsl))
          allocate(dump_ril(nobsl))
          allocate(dump_rjl(nobsl))
          allocate(dump_rkz(nobsl))
          dump_sets = 0
          dump_idx = 0
          dump_nn = 0
          dump_elms = 0
          dump_typs = 0
          dump_qc = 0
          dump_lon = 0.0_r_size
          dump_lat = 0.0_r_size
          dump_lev = 0.0_r_size
          dump_meta = 0.0_r_size
          dump_ri_global = 0.0_r_size
          dump_rj_global = 0.0_r_size
          dump_ril = 0.0_r_size
          dump_rjl = 0.0_r_size
          dump_rkz = undef
        end if

        if (slot_nobsg <= 0) then
          write (6, '(A)') ' -- no observations found in this time slot... do not need to read model data'
          if (nobsl_dump > 0) then
            deallocate(dump_sets, dump_idx, dump_nn, dump_elms, dump_typs, dump_qc, dump_lon, dump_lat, dump_lev, &
                       dump_meta, dump_ri_global, dump_rj_global, dump_ril, dump_rjl, dump_rkz)
          end if
          cycle
        end if

        if ( LOG_OUT ) then
          write (6, '(A,I10)') ' -- # obs in the slot = ', slot_nobsg
          write (6, '(A,I6,A,I6,A,I10)') ' -- # obs in the slot and processed by rank ', myrank, ' (subdomain #', myrank_d, ') = ', bsn(islot, myrank_d)
        end if

        call mpi_timer('', 2)

        call read_ens_history_iter(it, islot, v3dg, v2dg)

        if (obsop_dump_active()) then
          if ( RADAR_ADDITIVE_Y18 ) then
            call obsop_dump_state(it, im, islot, v3dg, v2dg, mv3dg(islot2,:,:,:,:), slope3dg(islot2,:,:))
          else
            call obsop_dump_state(it, im, islot, v3dg, v2dg)
          end if
        end if

        write (timer_str, '(A30,I4,A7,I4,A2)') 'obsope_cal:read_ens_history(t=', it, ', slot=', islot, '):'
        call mpi_timer(trim(timer_str), 2)

!$omp parallel do private(nn,n,iof,ril,rjl,rk,rkz,iloc)
        do nn = n1, n2
          iof = obsda%set(nn)
          n = obsda%idx(nn)

          call rij_g2l(myrank_d, obs(iof)%ri(n), obs(iof)%rj(n), ril, rjl)
          dump_ctx%obs_set  = obsda%set(nn)
          dump_ctx%obs_idx  = obsda%idx(nn)
          dump_ctx%elm      = obs(iof)%elm(n)
          dump_ctx%typ      = obs(iof)%typ(n)
          dump_ctx%stage    = 'obsop'
          dump_ctx%rank_global = myrank
          dump_ctx%rank_domain = myrank_d
          dump_ctx%ri       = ril
          dump_ctx%rj       = rjl
          dump_ctx%lon      = obs(iof)%lon(n)
          dump_ctx%lat      = obs(iof)%lat(n)
          dump_ctx%lev      = obs(iof)%lev(n)
          dump_ctx%rk       = 0.0_r_size

          if (nobsl_dump > 0) then
            iloc = nn - n1 + 1
            dump_sets(iloc) = obsda%set(nn)
            dump_idx(iloc) = obsda%idx(nn)
            dump_nn(iloc) = nn
            dump_elms(iloc) = obs(iof)%elm(n)
            dump_typs(iloc) = obs(iof)%typ(n)
            dump_lon(iloc) = obs(iof)%lon(n)
            dump_lat(iloc) = obs(iof)%lat(n)
            dump_lev(iloc) = obs(iof)%lev(n)
            dump_meta(:,iloc) = obs(iof)%meta
            dump_ri_global(iloc) = obs(iof)%ri(n)
            dump_rj_global(iloc) = obs(iof)%rj(n)
            dump_ril(iloc) = ril
            dump_rjl(iloc) = rjl
          end if

          if (.not. USE_OBS(obs(iof)%typ(n))) then
            obsda%qc(nn) = iqc_otype
            cycle
          end if

          select case (OBS_IN_FORMAT(iof))
          !=====================================================================
          case (obsfmt_prepbufr)
          !---------------------------------------------------------------------
            call phys2ijk(v3dg(:,:,:,iv3dd_p), obs(iof)%elm(n), ril, rjl, obs(iof)%lev(n), rk, obsda%qc(nn), typ=obs(iof)%typ(n))
            if (obsda%qc(nn) == iqc_good) then
              dump_ctx%rk = rk
              call Trans_XtoY(obs(iof)%elm(n), ril, rjl, rk, &
                              obs(iof)%lon(n), obs(iof)%lat(n), v3dg, v2dg, obsda%val(nn), obsda%qc(nn), typ=obs(iof)%typ(n), dump_ctx=dump_ctx)
            end if
          !=====================================================================
          case (obsfmt_radar, obsfmt_radar_nc)
          !---------------------------------------------------------------------
            if (obs(iof)%lev(n) > RADAR_ZMAX) then
              obsda%qc(nn) = iqc_radar_vhi
              if (LOG_LEVEL >= 3) then
                write(6,'(A,F8.1,A,I5)') '[Warning] radar observation is too high: lev=', obs(iof)%lev(n), ', elem=', obs(iof)%elm(n)
              end if
            elseif (obs(iof)%lev(n) < RADAR_ZMIN) then
              obsda%qc(nn) = iqc_out_vlo
              if (LOG_LEVEL >= 3) then
                write(6,'(A,F8.1,A,I5)') '[Warning] radar observation is too low: lev=', obs(iof)%lev(n), ', elem=', obs(iof)%elm(n)
              end if
            else
              call phys2ijkz(v3dg(:,:,:,iv3dd_hgt), ril, rjl, obs(iof)%lev(n), rkz, obsda%qc(nn))
            end if
            if (obsda%qc(nn) == iqc_good) then
              dump_ctx%rk = rkz
              if ( RADAR_ADDITIVE_Y18 ) then
                call Trans_XtoY_radar(obs(iof)%elm(n), obs(iof)%meta(1), obs(iof)%meta(2), obs(iof)%meta(3), ril, rjl, rkz,       &
                                      obs(iof)%lon(n), obs(iof)%lat(n), obs(iof)%lev(n), v3dg, v2dg, obsda%val(nn), obsda%qc(nn), &
                                      mv3d=mv3dg(islot-SLOT_START+1,:,:,:,:), slope3d=slope3dg(islot-SLOT_START+1,:,:), &
                                      ref_add=obsda%pert(nn), dump_ctx=dump_ctx )
              else
                call Trans_XtoY_radar(obs(iof)%elm(n), obs(iof)%meta(1), obs(iof)%meta(2), obs(iof)%meta(3), ril, rjl, rkz, &
                                      obs(iof)%lon(n), obs(iof)%lat(n), obs(iof)%lev(n), v3dg, v2dg, obsda%val(nn), obsda%qc(nn), &
                                      dump_ctx=dump_ctx)
              endif
            end if
            if (obsda%qc(nn) == iqc_ref_low) obsda%qc(nn) = iqc_good ! when process the observation operator, we don't care if reflectivity is too small
            if (nobsl_dump > 0) then
              iloc = nn - n1 + 1
              dump_rkz(iloc) = rkz
            end if

            if (RADAR_PQV) then
              call itpl_3d( v3dg(:,:,:,iv3dd_p), rkz, ril, rjl, obsda%pm(nn) )
              call itpl_3d( v3dg(:,:,:,iv3dd_t), rkz, ril, rjl, obsda%tm(nn) )
              call itpl_3d( v3dg(:,:,:,iv3dd_q), rkz, ril, rjl, obsda%qv(nn) )
            end if

            !!!!!! may not need to do this at this stage !!!!!!
            !if (obs(iof)%elm(n) == id_radar_ref_obs) then
            !  obsda%val(nn) = 10.0d0 * log10(obsda%val(nn))
            !end if
            !!!!!!
          end select


        end do ! [ nn = n1, n2 ]
!$omp end parallel do
 
        write (timer_str, '(A30,I4,A7,I4,A2)') 'obsope_cal:obsope_step_2   (t=', it, ', slot=', islot, '):'
        call mpi_timer(trim(timer_str), 2)

        if (nobsl_dump > 0) then
          dump_qc = obsda%qc(n1:n2)
          call obsop_dump_slot(it, im, islot, dump_sets, dump_idx, dump_nn, dump_elms, dump_typs, &
                               dump_lon, dump_lat, dump_lev, dump_meta, dump_ri_global, dump_rj_global, &
                               dump_ril, dump_rjl, dump_rkz, dump_qc, n1, n2)
          deallocate(dump_sets, dump_idx, dump_nn, dump_elms, dump_typs, dump_qc, dump_lon, dump_lat, dump_lev, &
                     dump_meta, dump_ri_global, dump_rj_global, dump_ril, dump_rjl, dump_rkz)
        end if
      end do ! [ islot = SLOT_START, SLOT_END ]

      call mpi_timer('', 2)

      ! Write obsda data to files if OBSDA_OUT = .true.
      ! 
      if (OBSDA_OUT) then
!        write (6,'(A,I6.6,A,I4.4,A,I6.6)') 'MYRANK ',myrank,' is writing observations for member ', &
!              im, ', subdomain id #', myrank_d
        if (im <= MEMBER) then
          obsdafile = OBSDA_OUT_BASENAME
          call filename_replace_mem(obsdafile, im)
        else if (im == mmean) then
          obsdafile = OBSDA_MEAN_OUT_BASENAME
        else if (im == mmdet) then
          obsdafile = OBSDA_MDET_OUT_BASENAME
        end if
        write (obsda_suffix(2:7),'(I6.6)') myrank_d
        write (6,'(A,I6.6,2A)') 'MYRANK ', myrank,' is writing an obsda file ', trim(obsdafile)//obsda_suffix
        call write_obs_da(trim(obsdafile)//obsda_suffix,obsda,0)

        write (timer_str, '(A30,I4,A2)') 'obsope_cal:write_obs_da    (t=', it, '):'
        call mpi_timer(trim(timer_str), 2)
      end if

      ! Prepare variables that will need to be communicated if obsda_return is given
      ! 
      if (present(obsda_return)) then
        if ( RADAR_PQV ) then
          call obs_da_value_partial_reduce_iter(obsda_return, it, 1, nobs, obsda%val, obsda%qc, qv=obsda%qv, tm=obsda%tm, pm=obsda%pm )
        elseif ( RADAR_ADDITIVE_Y18 ) then
          call obs_da_value_partial_reduce_iter(obsda_return, it, 1, nobs, obsda%val, obsda%qc, pert=obsda%pert)
        else
          call obs_da_value_partial_reduce_iter(obsda_return, it, 1, nobs, obsda%val, obsda%qc )
        endif

        write (timer_str, '(A30,I4,A2)') 'obsope_cal:partial_reduce  (t=', it, '):'
        call mpi_timer(trim(timer_str), 2)
      end if ! [ present(obsda_return) ]

    end if ! [ (im >= 1 .and. im <= MEMBER) .or. im == mmdetin ]
  end do ! [ it = 1, nitmax ]

  if ( RADAR_ADDITIVE_Y18 ) then
    deallocate ( mv3dg, mv2dg, mdbz3dg )
    deallocate ( slope3dg )
  endif

  deallocate ( v3dg, v2dg )
  deallocate ( bsn, bsna )
  call obs_da_value_deallocate(obsda)

  return
end subroutine obsope_cal

SUBROUTINE obsop_dump_state(iter, mem, islot, v3dg, v2dg, mv3d_slot, slope3d_slot)
  integer, intent(in) :: iter, mem, islot
  real(r_size), intent(in) :: v3dg(:,:,:,:)
  real(r_size), intent(in) :: v2dg(:,:,:)
  real(r_size), intent(in), optional :: mv3d_slot(:,:,:,:)
  real(r_size), intent(in), optional :: slope3d_slot(:,:)
  character(len=filelenmax) :: stage_dir
  character(len=filelenmax) :: file_path

  if (.not. obsop_dump_active()) return
  call obsop_prepare_base()
  stage_dir = obsop_stage_directory('state', mem, islot, iter)
  call obsop_write_real4d(append_dir(stage_dir, 'v3dg'//obsop_dump_suffix), v3dg)
  call obsop_write_real3d(append_dir(stage_dir, 'v2dg'//obsop_dump_suffix), v2dg)
  if (present(mv3d_slot)) then
    call obsop_write_real4d(append_dir(stage_dir, 'mv3dg'//obsop_dump_suffix), mv3d_slot)
  end if
  if (present(slope3d_slot)) then
    call obsop_write_real_matrix(append_dir(stage_dir, 'slope3dg'//obsop_dump_suffix), slope3d_slot)
  end if
  file_path = append_dir(stage_dir, 'stage_meta.txt')
  call obsop_write_stage_meta(file_path, iter, islot, mem, -1, -1, -1)
END SUBROUTINE obsop_dump_state

SUBROUTINE obsop_dump_slot(iter, mem, islot, obs_sets, obs_idxs, obs_nn, obs_elms, obs_typs, &
                           obs_lon, obs_lat, obs_lev, obs_meta, obs_ri_global, obs_rj_global, &
                           obs_ril, obs_rjl, obs_rkz, obs_qc, obs_n1, obs_n2)
  integer, intent(in) :: iter, mem, islot
  integer, intent(in) :: obs_sets(:), obs_idxs(:), obs_nn(:), obs_elms(:), obs_typs(:)
  integer, intent(in) :: obs_qc(:)
  real(r_size), intent(in) :: obs_lon(:), obs_lat(:), obs_lev(:)
  real(r_size), intent(in) :: obs_meta(:,:)
  real(r_size), intent(in) :: obs_ri_global(:), obs_rj_global(:)
  real(r_size), intent(in) :: obs_ril(:), obs_rjl(:), obs_rkz(:)
  integer, intent(in) :: obs_n1, obs_n2
  integer :: nobs
  character(len=filelenmax) :: stage_dir
  character(len=filelenmax) :: file_path

  if (.not. obsop_dump_active()) return
  nobs = size(obs_sets)
  if (nobs <= 0) return
  call obsop_prepare_base()
  stage_dir = obsop_stage_directory('obs', mem, islot, iter)

  call obsop_write_integer_vector(append_dir(stage_dir, 'set'//obsop_dump_suffix), obs_sets)
  call obsop_write_integer_vector(append_dir(stage_dir, 'idx'//obsop_dump_suffix), obs_idxs)
  call obsop_write_integer_vector(append_dir(stage_dir, 'nn'//obsop_dump_suffix), obs_nn)
  call obsop_write_integer_vector(append_dir(stage_dir, 'elm'//obsop_dump_suffix), obs_elms)
  call obsop_write_integer_vector(append_dir(stage_dir, 'typ'//obsop_dump_suffix), obs_typs)
  call obsop_write_integer_vector(append_dir(stage_dir, 'qc'//obsop_dump_suffix), obs_qc)
  call obsop_write_real_vector(append_dir(stage_dir, 'lon'//obsop_dump_suffix), obs_lon)
  call obsop_write_real_vector(append_dir(stage_dir, 'lat'//obsop_dump_suffix), obs_lat)
  call obsop_write_real_vector(append_dir(stage_dir, 'lev'//obsop_dump_suffix), obs_lev)
  call obsop_write_real_vector(append_dir(stage_dir, 'ri_global'//obsop_dump_suffix), obs_ri_global)
  call obsop_write_real_vector(append_dir(stage_dir, 'rj_global'//obsop_dump_suffix), obs_rj_global)
  call obsop_write_real_vector(append_dir(stage_dir, 'ril'//obsop_dump_suffix), obs_ril)
  call obsop_write_real_vector(append_dir(stage_dir, 'rjl'//obsop_dump_suffix), obs_rjl)
  call obsop_write_real_vector(append_dir(stage_dir, 'rkz'//obsop_dump_suffix), obs_rkz)
  call obsop_write_real_matrix(append_dir(stage_dir, 'meta'//obsop_dump_suffix), obs_meta)

  file_path = append_dir(stage_dir, 'stage_meta.txt')
  call obsop_write_stage_meta(file_path, iter, islot, mem, nobs, obs_n1, obs_n2)
END SUBROUTINE obsop_dump_slot

SUBROUTINE obsop_prepare_base()
  character(len=filelenmax) :: parent
  if (obsop_dir_ready) return
  parent = obsop_parent_dir(obsop_trim_dir(LETKF_INPUT_DUMP_DIR))
  obsop_base_dir = append_dir(parent, obsop_dump_dirname)
  call obsop_mkdir(obsop_base_dir)
  obsop_dir_ready = .true.
END SUBROUTINE obsop_prepare_base

FUNCTION obsop_stage_directory(section, mem, islot, iter) RESULT(dir_out)
  character(len=*), intent(in) :: section
  integer, intent(in) :: mem, islot, iter
  character(len=filelenmax) :: dir_out
  character(len=32) :: stage_tag
  write(stage_tag,'(A2,I4.4,A5,I4.4)') 'it', iter, '_slot', islot
  dir_out = append_dir(obsop_base_dir, trim(section))
  dir_out = append_dir(dir_out, obsop_domain_tag())
  dir_out = append_dir(dir_out, obsop_member_tag(mem))
  dir_out = append_dir(dir_out, trim(stage_tag))
  call obsop_mkdir(dir_out)
END FUNCTION obsop_stage_directory

FUNCTION obsop_domain_tag() RESULT(tag)
  character(len=8) :: tag
  write(tag,'(A2,I6.6)') 'pe', myrank_d
END FUNCTION obsop_domain_tag

FUNCTION obsop_member_tag(mem_index) RESULT(tag)
  integer, intent(in) :: mem_index
  character(len=memflen+3) :: tag
  character(len=memflen) :: core
  if (mem_index >= 1 .and. mem_index <= MEMBER) then
    write(core,'(I4.4)') mem_index
  else if (mem_index == mmean) then
    core = memf_mean
  else if (mem_index == mmdet) then
    core = memf_mdet
  else if (mem_index == mmgue) then
    core = memf_mgue
  else
    write(core,'(I4.4)') mem_index
  end if
  tag = 'mem'//trim(core)
END FUNCTION obsop_member_tag

SUBROUTINE obsop_mkdir(path)
  character(len=*), intent(in) :: path
  if (myrank == 0) then
    call execute_command_line('mkdir -p ' // trim(path))
  end if
END SUBROUTINE obsop_mkdir

FUNCTION obsop_trim_dir(path_in) RESULT(path_out)
  character(len=*), intent(in) :: path_in
  character(len=filelenmax) :: path_out
  path_out = adjustl(path_in)
  if (len_trim(path_out) == 0) path_out = 'letkf_dump'
END FUNCTION obsop_trim_dir

FUNCTION obsop_parent_dir(dir_in) RESULT(dir_out)
  character(len=*), intent(in) :: dir_in
  character(len=filelenmax) :: dir_out
  integer :: i, last_sep, end_pos
  dir_out = adjustl(dir_in)
  end_pos = len_trim(dir_out)
  if (end_pos <= 0) then
    dir_out = '.'
    return
  end if
  do while (end_pos > 1 .and. (dir_out(end_pos:end_pos) == '/' .or. dir_out(end_pos:end_pos) == '\'))
    end_pos = end_pos - 1
  end do
  last_sep = 0
  do i = 1, end_pos
    if (dir_out(i:i) == '/' .or. dir_out(i:i) == '\') last_sep = i
  end do
  if (last_sep <= 0) then
    dir_out = '.'
  else if (last_sep == 1 .and. dir_out(1:1) == '/') then
    dir_out = '/'
  else
    dir_out = dir_out(1:last_sep-1)
  end if
END FUNCTION obsop_parent_dir

FUNCTION append_dir(parent, child) RESULT(path)
  character(len=*), intent(in) :: parent
  character(len=*), intent(in) :: child
  character(len=filelenmax) :: path
  path = trim(parent)//'/'//trim(child)
END FUNCTION append_dir

SUBROUTINE obsop_write_real_vector(path, data)
  character(len=*), intent(in) :: path
  real(r_size), intent(in) :: data(:)
  integer :: unit
  integer(int32) :: nd, dims(1)
  nd = 1
  dims(1) = size(data)
  open(newunit=unit, file=trim(path), status='replace', access='stream', form='unformatted')
  write(unit) nd
  write(unit) dims
  if (dims(1) > 0) write(unit) data
  close(unit)
END SUBROUTINE obsop_write_real_vector

SUBROUTINE obsop_write_integer_vector(path, data)
  character(len=*), intent(in) :: path
  integer, intent(in) :: data(:)
  integer :: unit
  integer(int32) :: nd, dims(1)
  nd = 1
  dims(1) = size(data)
  open(newunit=unit, file=trim(path), status='replace', access='stream', form='unformatted')
  write(unit) nd
  write(unit) dims
  if (dims(1) > 0) write(unit) data
  close(unit)
END SUBROUTINE obsop_write_integer_vector

SUBROUTINE obsop_write_real_matrix(path, data)
  character(len=*), intent(in) :: path
  real(r_size), intent(in) :: data(:,:)
  integer :: unit
  integer(int32) :: nd, dims(2)
  nd = 2
  dims = (/ size(data,1), size(data,2) /)
  open(newunit=unit, file=trim(path), status='replace', access='stream', form='unformatted')
  write(unit) nd
  write(unit) dims
  if (dims(1) > 0 .and. dims(2) > 0) write(unit) data
  close(unit)
END SUBROUTINE obsop_write_real_matrix

SUBROUTINE obsop_write_real3d(path, data)
  character(len=*), intent(in) :: path
  real(r_size), intent(in) :: data(:,:,:)
  integer :: unit
  integer(int32) :: nd, dims(3)
  nd = 3
  dims = (/ size(data,1), size(data,2), size(data,3) /)
  open(newunit=unit, file=trim(path), status='replace', access='stream', form='unformatted')
  write(unit) nd
  write(unit) dims
  if (product(dims) > 0) write(unit) data
  close(unit)
END SUBROUTINE obsop_write_real3d

SUBROUTINE obsop_write_real4d(path, data)
  character(len=*), intent(in) :: path
  real(r_size), intent(in) :: data(:,:,:,:)
  integer :: unit
  integer(int32) :: nd, dims(4)
  nd = 4
  dims = (/ size(data,1), size(data,2), size(data,3), size(data,4) /)
  open(newunit=unit, file=trim(path), status='replace', access='stream', form='unformatted')
  write(unit) nd
  write(unit) dims
  if (product(dims) > 0) write(unit) data
  close(unit)
END SUBROUTINE obsop_write_real4d

SUBROUTINE obsop_write_stage_meta(path, iter, islot, mem, nobs, n1, n2)
  character(len=*), intent(in) :: path
  integer, intent(in) :: iter, islot, mem
  integer, intent(in) :: nobs, n1, n2
  integer :: unit
  open(newunit=unit, file=trim(path), status='replace')
  write(unit,'(A,I0)') 'iter=', iter
  write(unit,'(A,I0)') 'slot=', islot
  write(unit,'(A,I0)') 'member=', mem
  write(unit,'(A,I0)') 'nobs=', nobs
  write(unit,'(A,I0)') 'n1=', n1
  write(unit,'(A,I0)') 'n2=', n2
  close(unit)
END SUBROUTINE obsop_write_stage_meta

LOGICAL FUNCTION obsop_dump_active()
  obsop_dump_active = LETKF_INPUT_DUMP .and. (obsop_dump_target_rank < 0 .or. myrank == obsop_dump_target_rank)
END FUNCTION obsop_dump_active

!-----------------------------------------------------------------------
! Observation generator calculation
!-----------------------------------------------------------------------
SUBROUTINE obsmake_cal(obs)
  IMPLICIT NONE

  TYPE(obs_info),INTENT(INOUT) :: obs(OBS_IN_NUM)
  REAL(r_size),ALLOCATABLE :: v3dg(:,:,:,:)
  REAL(r_size),ALLOCATABLE :: v2dg(:,:,:)

  integer :: islot,proc
  integer :: n,nslot,nobs,nobs_slot,ierr,iqc,iof
  integer :: nobsmax,nobsall
  real(r_size) :: rig,rjg,ril,rjl,rk,rkz
  real(r_size) :: slot_lb,slot_ub
  real(r_size),allocatable :: bufr(:)
  real(r_size),allocatable :: error(:)

  INTEGER :: ns 

!-----------------------------------------------------------------------

  write (6,'(A,I6.6,A,I6.6)') 'MYRANK ', myrank, ' is processing subdomain id #', myrank_d

  allocate ( v3dg (nlevh,nlonh,nlath,nv3dd) )
  allocate ( v2dg (nlonh,nlath,nv2dd) )

  do iof = 1, OBS_IN_NUM
    obs(iof)%dat = 0.0d0
  end do

  nobs = 0
  do islot = SLOT_START, SLOT_END
    slot_lb = (real(islot-SLOT_BASE,r_size) - 0.5d0) * SLOT_TINTERVAL
    slot_ub = (real(islot-SLOT_BASE,r_size) + 0.5d0) * SLOT_TINTERVAL
    write (6,'(A,I3,A,F9.1,A,F9.1,A)') 'Slot #', islot-SLOT_START+1, ': time window (', slot_lb, ',', slot_ub, '] sec'

    call read_ens_history_iter(1,islot,v3dg,v2dg)

    do iof = 1, OBS_IN_NUM
        nslot = 0
        nobs_slot = 0
        do n = 1, obs(iof)%nobs

          if (obs(iof)%dif(n) > slot_lb .and. obs(iof)%dif(n) <= slot_ub) then
            nslot = nslot + 1

            call phys2ij(obs(iof)%lon(n),obs(iof)%lat(n),rig,rjg)
            call rij_rank_g2l(rig,rjg,proc,ril,rjl)

  !          if (myrank_d == 0) then
  !            print *, proc, rig, rjg, ril, rjl
  !          end if

            if (proc < 0 .and. myrank_d == 0) then ! if outside of the global domain, processed by myrank_d == 0
              obs(iof)%dat(n) = undef
            end if

            if (myrank_d == proc) then
              nobs = nobs + 1
              nobs_slot = nobs_slot + 1

  !IF(NINT(elem(n)) == id_ps_obs) THEN
  !  CALL itpl_2d(v2d(:,:,iv2d_orog),ril,rjl,dz)
  !  rk = rlev(n) - dz
  !  IF(ABS(rk) > threshold_dz) THEN ! pressure adjustment threshold
  !    ! WRITE(6,'(A)') '* PS obs vertical adjustment beyond threshold'
  !    ! WRITE(6,'(A,F10.2,A,F6.2,A,F6.2,A)') '* dz=',rk,&
  !    ! & ', (lon,lat)=(',elon(n),',',elat(n),')'
  !    CYCLE
  !  END IF
  !END IF

              select case (OBS_IN_FORMAT(iof))
              !=================================================================
              case (obsfmt_prepbufr)
              !-----------------------------------------------------------------
                call phys2ijk(v3dg(:,:,:,iv3dd_p),obs(iof)%elm(n),ril,rjl,obs(iof)%lev(n),rk,iqc,typ=obs(iof)%typ(n))
                if (iqc == iqc_good) then
                  call Trans_XtoY(obs(iof)%elm(n),ril,rjl,rk, &
                                  obs(iof)%lon(n),obs(iof)%lat(n),v3dg,v2dg,obs(iof)%dat(n),iqc,typ=obs(iof)%typ(n))
                end if
              !=================================================================
              case (obsfmt_radar, obsfmt_radar_nc )
              !-----------------------------------------------------------------
                call phys2ijkz(v3dg(:,:,:,iv3dd_hgt),ril,rjl,obs(iof)%lev(n),rkz,iqc)
                if (iqc == iqc_good) then
                  call Trans_XtoY_radar(obs(iof)%elm(n),obs(iof)%meta(1),obs(iof)%meta(2),obs(iof)%meta(3),ril,rjl,rkz, &
                                        obs(iof)%lon(n),obs(iof)%lat(n),obs(iof)%lev(n),v3dg,v2dg,obs(iof)%dat(n),iqc)
 !!! For radar observation, when reflectivity value is too low, do not generate ref/vr observations
 !!! No consideration of the terrain blocking effects.....
                end if
              end select

              if (iqc == iqc_ref_low .and. int(obs(iof)%elm(n)) == id_radar_ref_zero_obs ) then
                obs(iof)%dat(n) = MIN_RADAR_REF_DBZ + LOW_REF_SHIFT  !!! not undef -> not rejected
              elseif (iqc /= iqc_ref_low .and. int(obs(iof)%elm(n)) == id_radar_ref_zero_obs ) then
                obs(iof)%dat(n) = undef
                iqc = iqc_ref_mem
              elseif (iqc /= iqc_good) then
                obs(iof)%dat(n) = undef
              end if

            end if ! [ myrank_d == proc ]

          end if ! [ obs%dif(n) > slot_lb .and. obs%dif(n) <= slot_ub ]

        end do ! [ n = 1, obs%nobs ]


    end do ! [ iof = 1, OBS_IN_NUM ]

    if ( LOG_OUT ) then
      write (6,'(3A,I10)') ' -- [', trim(OBS_IN_NAME(iof)), '] nobs in the slot = ', nslot
      write (6,'(3A,I6,A,I10)') ' -- [', trim(OBS_IN_NAME(iof)), '] nobs in the slot and processed by rank ', myrank, ' = ', nobs_slot
    end if

  end do ! [ islot = SLOT_START, SLOT_END ]

  deallocate ( v3dg, v2dg )

  if (myrank_d == 0) then
    nobsmax = 0
    nobsall = 0
    do iof = 1, OBS_IN_NUM
      if (obs(iof)%nobs > nobsmax) nobsmax = obs(iof)%nobs
      nobsall = nobsall + obs(iof)%nobs
    end do

    allocate ( bufr(nobsmax) )
    allocate ( error(nobsall) )

    call com_randn(nobsall, error) ! generate all random numbers at the same time
    ns = 0
  else
    allocate ( bufr(1) )
  end if

  do iof = 1, OBS_IN_NUM

    if (myrank_d == 0) then
      call MPI_REDUCE(obs(iof)%dat,bufr(1:obs(iof)%nobs),obs(iof)%nobs,MPI_r_size,MPI_SUM,0,MPI_COMM_d,ierr)
    else
      call MPI_REDUCE(obs(iof)%dat,bufr(1:1),obs(iof)%nobs,MPI_r_size,MPI_SUM,0,MPI_COMM_d,ierr)
    end if

    if (myrank_d == 0) then
      obs(iof)%dat = bufr(1:obs(iof)%nobs)

      do n = 1, obs(iof)%nobs
        select case(obs(iof)%elm(n))
        case(id_u_obs)
          obs(iof)%err(n) = OBSERR_U
        case(id_v_obs)
          obs(iof)%err(n) = OBSERR_V
        case(id_t_obs,id_tv_obs)
          obs(iof)%err(n) = OBSERR_T
        case(id_q_obs)
          obs(iof)%err(n) = OBSERR_Q
        case(id_rh_obs)
          obs(iof)%err(n) = OBSERR_RH
        case(id_ps_obs)
          obs(iof)%err(n) = OBSERR_PS
        case(id_radar_ref_obs,id_radar_ref_zero_obs)
          obs(iof)%err(n) = OBSERR_RADAR_REF
        case(id_radar_vr_obs)
          obs(iof)%err(n) = OBSERR_RADAR_VR
        case default
          write(6,'(A)') '[Warning] skip assigning observation error (unsupported observation type)'
        end select

        if (obs(iof)%dat(n) /= undef .and. obs(iof)%err(n) /= undef) then
          obs(iof)%dat(n) = obs(iof)%dat(n) + obs(iof)%err(n) * error(ns+n)
          if (any(obs(iof)%elm(n) == (/id_t_obs,id_tv_obs,id_q_obs,id_rh_obs,id_ps_obs/))) then
            obs(iof)%dat(n) = max(obs(iof)%dat(n), 0.0_r_size)
          end if
        end if

        if (int(obs(iof)%elm(n)) == id_radar_ref_obs .and. obs(iof)%dat(n) /= undef ) then
          obs(iof)%dat(n) = 10.0_r_size** (obs(iof)%dat(n) / 10.0_r_size ) !!! dBZ -> original unit
        elseif (int(obs(iof)%elm(n)) == id_radar_ref_zero_obs .and. obs(iof)%dat(n) /= undef ) then
          obs(iof)%dat(n) = 10.0_r_size** ((MIN_RADAR_REF_DBZ + LOW_REF_SHIFT ) / 10.0_r_size ) !!! dBZ -> original unit
        end if

!print *, '######', obs%elm(n), obs%dat(n)
      end do ! [ n = 1, obs(iof)%nobs ]

      ns = ns + obs(iof)%nobs
    end if ! [ myrank_d == 0 ]

  end do ! [ iof = 1, OBS_IN_NUM ]

  if (myrank_d == 0) then
    deallocate ( bufr )
    deallocate ( error )

    call write_obs_all(obs, missing=.false., file_suffix='.out') ! only at the head node
  else
    deallocate ( bufr )
  end if

end subroutine obsmake_cal

!-------------------------------------------------------------------------------
! Model-to-observation simulator calculation
!-------------------------------------------------------------------------------
subroutine obssim_cal(v3dgh, v2dgh, v3dgsim, v2dgsim, stggrd)
  use scale_atmos_grid_cartesC, only: &
      CX => ATMOS_GRID_CARTESC_CX, &
      CY => ATMOS_GRID_CARTESC_CY, &
      DX, DY
  use scale_atmos_grid_cartesC_index, only: &
      IHALO, JHALO, KHALO
  use scale_mapprojection, only: &
      MAPPROJECTION_xy2lonlat

  implicit none

  real(r_size), intent(in) :: v3dgh(nlevh,nlonh,nlath,nv3dd)
  real(r_size), intent(in) :: v2dgh(nlonh,nlath,nv2dd)
  real(r_size), intent(out) :: v3dgsim(nlev,nlon,nlat,OBSSIM_NUM_3D_VARS)
  real(r_size), intent(out) :: v2dgsim(nlon,nlat,OBSSIM_NUM_2D_VARS)
  integer, intent(in), optional :: stggrd

  integer :: i, j, k, iv3dsim, iv2dsim
  real(r_size) :: ri, rj, rk
  real(r_size) :: lon, lat, lev
  real(r_size) :: tmpobs
  real(RP) :: lon_RP, lat_RP
  integer :: tmpqc

!-------------------------------------------------------------------------------

  write (6,'(A,I6.6,A,I6.6)') 'MYRANK ', myrank, ' is processing subdomain id #', myrank_d

  do j = 1, nlat
    rj = real(j + JHALO, r_size)

    do i = 1, nlon
      ri = real(i + IHALO, r_size)
      call MAPPROJECTION_xy2lonlat( real(ri - 1.0_r_size, kind=RP)*DX + CX(1), &
                                    real(rj - 1.0_r_size, kind=RP)*DY + CY(1), &
                                    lon_RP, lat_RP )
      lon = real(lon_RP, kind=r_size)*rad2deg
      lat = real(lat_RP, kind=r_size)*rad2deg

      do k = 1, nlev
        rk = real(k + KHALO, r_size)

        do iv3dsim = 1, OBSSIM_NUM_3D_VARS
          select case (OBSSIM_3D_VARS_LIST(iv3dsim))
          case (id_radar_ref_obs, id_radar_ref_zero_obs, id_radar_vr_obs, id_radar_prh_obs)
            lev = v3dgh(k+KHALO, i+IHALO, j+JHALO, iv3dd_hgt)
            call Trans_XtoY_radar(OBSSIM_3D_VARS_LIST(iv3dsim), OBSSIM_RADAR_LON, OBSSIM_RADAR_LAT, OBSSIM_RADAR_Z, ri, rj, rk, &
                                  lon, lat, lev, v3dgh, v2dgh, tmpobs, tmpqc, stggrd)
            if (tmpqc == iqc_ref_low) tmpqc = iqc_good ! when process the observation operator, we don't care if reflectivity is too small
          case default
            call Trans_XtoY(OBSSIM_3D_VARS_LIST(iv3dsim), ri, rj, rk, &
                            lon, lat, v3dgh, v2dgh, tmpobs, tmpqc, stggrd)
          end select

          if (tmpqc == 0) then
            v3dgsim(k,i,j,iv3dsim) = real(tmpobs, r_sngl)
          else
            v3dgsim(k,i,j,iv3dsim) = real(undef, r_sngl)
          end if
        end do ! [ iv3dsim = 1, OBSSIM_NUM_3D_VARS ]

        ! 2D observations calculated when k = 1
        if (k == 1) then
          do iv2dsim = 1, OBSSIM_NUM_2D_VARS
            select case (OBSSIM_2D_VARS_LIST(iv2dsim))
            case default
              call Trans_XtoY(OBSSIM_2D_VARS_LIST(iv2dsim), ri, rj, rk, &
                              lon, lat, v3dgh, v2dgh, tmpobs, tmpqc, stggrd)
            end select

            if (tmpqc == 0) then
              v2dgsim(i,j,iv2dsim) = real(tmpobs, r_sngl)
            else
              v2dgsim(i,j,iv2dsim) = real(undef, r_sngl)
            end if
          end do ! [ iv2dsim = 1, OBSSIM_NUM_2D_VARS ]
        end if ! [ k == 1 ]

      end do ! [ k = 1, nlev ]

    end do ! [ i = 1, nlon ]

  end do ! [ j = 1, nlat ]

!-------------------------------------------------------------------------------

end subroutine obssim_cal

!!!!!! it is not good to open/close a file many times for different steps !!!!!!
!-------------------------------------------------------------------------------
! Write the subdomain model data into a single GrADS file
!-------------------------------------------------------------------------------
subroutine write_grd_mpi(filename, nv3dgrd, nv2dgrd, step, v3d, v2d)
  implicit none
  character(*), intent(in) :: filename
  integer, intent(in) :: nv3dgrd
  integer, intent(in) :: nv2dgrd
  integer, intent(in) :: step
  real(r_size), intent(in) :: v3d(nlev,nlon,nlat,nv3dgrd)
  real(r_size), intent(in) :: v2d(nlon,nlat,nv2dgrd)

  real(r_sngl) :: bufs4(nlong,nlatg)
  real(r_sngl) :: bufr4(nlong,nlatg)
  integer :: iunit, iolen
  integer :: k, n, irec, ierr
  integer :: proc_i, proc_j
  integer :: ishift, jshift

  call rank_1d_2d(myrank_d, proc_i, proc_j)
  ishift = proc_i * nlon
  jshift = proc_j * nlat

  if (myrank_d == 0) then
    iunit = 55
    inquire (iolength=iolen) iolen
    open (iunit, file=trim(filename), form='unformatted', access='direct', &
          status='unknown', recl=nlong*nlatg*iolen)
    irec = (nlev * nv3dgrd + nv2dgrd) * (step-1)
  end if

  do n = 1, nv3dgrd
    do k = 1, nlev
      bufs4(:,:) = 0.0
      bufs4(1+ishift:nlon+ishift, 1+jshift:nlat+jshift) = real(v3d(k,:,:,n), r_sngl)
      call MPI_REDUCE(bufs4, bufr4, nlong*nlatg, MPI_REAL, MPI_SUM, 0, MPI_COMM_d, ierr)
      if (myrank_d == 0) then
        irec = irec + 1
        write (iunit, rec=irec) bufr4
      end if
    end do
  end do

  do n = 1, nv2dgrd
    bufs4(:,:) = 0.0
    bufs4(1+ishift:nlon+ishift, 1+jshift:nlat+jshift) = real(v2d(:,:,n), r_sngl)
    call MPI_REDUCE(bufs4, bufr4, nlong*nlatg, MPI_REAL, MPI_SUM, 0, MPI_COMM_d, ierr)
    if (myrank_d == 0) then
      irec = irec + 1
      write (iunit, rec=irec) bufr4
    end if
  end do

  if (myrank_d == 0) then
    close (iunit)
  end if

  return
end subroutine write_grd_mpi

!=======================================================================

END MODULE obsope_tools
