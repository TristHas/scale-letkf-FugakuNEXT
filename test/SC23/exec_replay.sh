#!/bin/sh
#
#PJM -g rccs-sdt
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -L "rscgrp=small"
#PJM -L "node=15"
#PJM -L "elapse=00:10:00"
#PJM --mpi "max-proc-per-node=4"
#PJM -j
#PJM -s

export OMP_NUM_THREADS=12
export FORT90L=-Wl,-T
export PLE_MPI_STD_EMPTYFILE=off
export OMP_WAIT_POLICY=active

/usr/bin/time -f "\n[%C]\nreal %E user %U sys %S\n" -o timing.log -a \
  mpiexec -n 60 --output-filename log/NOUT_letkf_replay --tag-output \
  bin/letkf_replay conf/letkf_20210730060030.conf
