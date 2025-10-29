NC_HASH=`spack find -lx netcdf-c | grep netcdf-c | awk '{print $1}'`
NF_HASH=`spack find -lx netcdf-fortran | grep netcdf-fortran | awk '{print $1}'`
HDF_HASH=`spack find -l --deps /${NC_HASH} | grep hdf5 | awk '{print $1}'`
SCALE_HDF=`spack location --install-dir /${HDF_HASH}`
SCALE_NETCDF_C=`spack location --install-dir /${NC_HASH}`
SCALE_NETCDF_F=`spack location --install-dir /${NF_HASH}`
SCALE_PNETCDF=`spack location --install-dir parallel-netcdf`

export SCALE_DB="`pwd`/scale_database"
export SCALE_NETCDF_INCLUDE="-I${SCALE_NETCDF_C}/include -I${SCALE_NETCDF_F}/include -I${SCALE_PNETCDF}/include"
export SCALE_NETCDF_LIBS="-L${SCALE_NETCDF_C}/lib -L${SCALE_NETCDF_F}/lib -L${SCALE_HDF}/lib -L${SCALE_PNETCDF}/lib -lpnetcdf -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -lmpi_cxx"

OB=$(spack location -i openblas)
export SCALE_ENABLE_MATHLIB=T
export SCALE_MATHLIB_LIBS="-L${OB}/lib -lopenblas"

export LD_LIBRARY_PATH=\
/home/tristan/workspace/scale_letkf/spack/opt/spack/linux-zen2/netcdf-c-4.9.3-yhzaygyfyfhqhyzps6mfj6g6zfmztvk6/lib:\
/home/tristan/workspace/scale_letkf/spack/opt/spack/linux-zen2/netcdf-fortran-4.6.2-lthbjc2parfzvofptyrq5ztrn4xewppu/lib:\
/home/tristan/workspace/scale_letkf/spack/opt/spack/linux-zen2/hdf5-1.14.6-skiwbt33ejf5nim24hapkdvnpczeal6n/lib:\
/home/tristan/workspace/scale_letkf/spack/opt/spack/linux-zen2/parallel-netcdf-1.14.1-negcdbtbhsfomazpwhbgwhz5yzel2vkv/lib:\
${LD_LIBRARY_PATH}

export SCALE_SYS="Linux64-gnu-ompi"
