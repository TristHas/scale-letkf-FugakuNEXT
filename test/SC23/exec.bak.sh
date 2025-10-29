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
#export FLIB_BARRIER=HARD
#export LD_LIBRARY_PATH=/lib64:/opt/FJSVxtclanga/tcsds-mpi-latest/lib64:/opt/FJSVxtclanga/tcsds-latest/lib64:`cat /home/apps/oss/scale/llio.list | sed 's:\(.*/lib\)/.*:\1:' | uniq | sed -z 's/\n/:/g'`

 mpiexec -n 20 --output-filename log/NOUT_scale-rm_pp_ens --tag-output bin/scale-rm_pp_ens conf/scale-rm_pp_ens_20210730060000.conf
mpiexec -n 60 --output-filename log/NOUT_scale-rm_init_ens --tag-output bin/scale-rm_init_ens conf/scale-rm_init_ens_20210730060000.conf
mpiexec -n 60 --output-filename log/NOUT_scale-rm_ens --tag-output bin/scale-rm_ens conf/scale-rm_ens_20210730060000.conf
mpiexec -n 60 --output-filename log/NOUT_letkf --tag-output bin/letkf conf/letkf_20210730060030.conf
