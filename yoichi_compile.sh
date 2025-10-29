spack install parallel-netcdf
spack install netcdf-c
spack install netcdf-fortran
spack install openblas


export SCALE_SYS="Linux64-gnu-ompi"
sed -i 's/\<_FillValue\>\/"_FillValue"/g'   scale/scalelib/src/file/scale_file_netcdf.c
cd scale/scale-rm/src
make -j
cd ../../..
sed -i 's/(3A,I)/(3A,I0)/g' scale/scale-letkf/scale/common/common_obs_scale.f90
cd scale/scale-letkf/scale/
make -j
