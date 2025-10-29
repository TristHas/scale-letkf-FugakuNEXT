include ../configure

PROGS   = letkf efso

API_TARGET = libletkf_api.so
API_OBJS   = letkf_c_api.o

OBJS    = letkf_obs.o efso_tools.o letkf_tools.o

COMMON_DIR = ../common

COMMON_OBJS = $(COMMON_DIR)/SFMT.o           \
              $(COMMON_DIR)/common.o         \
              $(COMMON_DIR)/common_mpi.o     \
              $(COMMON_DIR)/$(common_mtx).o     \
              $(COMMON_DIR)/common_sort.o    \
              $(COMMON_DIR)/common_rand.o    \
              $(COMMON_DIR)/common_letkf.o   \
              $(COMMON_DIR)/common_ncio.o    \
              $(COMMON_DIR)/common_scale.o     \
              $(COMMON_DIR)/common_obs_scale.o \
              $(COMMON_DIR)/common_mpi_scale.o \
              $(COMMON_DIR)/common_nml.o

OBS_DIR = ../obs

OBS_OBJS = $(OBS_DIR)/obsope_tools.o

INCS    = $(SCALE_INC) $(NETCDF_INC)

LIBS    = $(SCALE_LIB) $(NETCDF_LIB) $(LAPACK_LIB) $(KEVD_LIB)


.PHONY:	all clean FORCE

all:	$(PROGS) $(API_TARGET)

letkf:	letkf.o $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) $(FMCMODEL) -o letkf letkf.o \
	$(COMMON_OBJS) $(OBS_OBJS) $(OBJS) $(SCALE_RM_OBJS) $(LIBS)

letkf.o:	letkf.f90 $(OBJS) $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) -I$(OBS_DIR) $(INCS)

letkf_obs.o:	letkf_obs.f90 $(COMMON_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) $(INCS)
letkf_tools.o:	letkf_tools.f90 letkf_obs.o efso_tools.o $(COMMON_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) $(INCS)

letkf_c_api.o:	letkf_c_api.f90 letkf_obs.o letkf_tools.o $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) -I$(OBS_DIR) $(INCS)

$(API_TARGET):	$(API_OBJS) $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) $(FMCMODEL) -shared -o $@ $(API_OBJS) \
		$(COMMON_OBJS) $(OBS_OBJS) $(OBJS) $(SCALE_RM_OBJS) $(LIBS)

$(OBS_DIR):	FORCE
	$(MAKE) -C $@ obsope_tools.o

efso:	efso.o $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) $(FMCMODEL) -o efso efso.o \
	$(COMMON_OBJS) $(OBS_OBJS) $(OBJS) $(SCALE_RM_OBJS) $(LIBS)

efso.o:		efso.f90 $(OBJS) $(COMMON_DIR) $(OBS_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) -I$(OBS_DIR) $(INCS)

efso_tools.o:	efso_tools.f90 letkf_obs.o $(COMMON_DIR)
	$(FC) $(FOPTS) -c $< -I$(COMMON_DIR) $(INCS)

clean:
	rm -f *.o *.mod *.lst $(PROGS) $(API_TARGET)

FORCE:
