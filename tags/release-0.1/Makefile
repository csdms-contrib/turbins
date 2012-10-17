# Number of processors to be used. Maximum 2 
THREADS = 2

TOP = $(PWD)
SRC_DIR = $(TOP)/src/
SRC_INCLUDE = $(SRC_DIR)/Include

# Make sure that the Petsc-dir is correct. 
PETSC_DIR  = /home/mmnasr/Petsc/petsc-3.1-p8
# Architeture of the operating system
PETSC_ARCH = linux-gnu-hdf5-o3

# C-Compiler
CC = /curc/tools/free/redhat_5_x86_64/openmpi-1.4.3_gcc-4.5.2_torque-2.5.8_ib/bin/mpicc  -Wall -Wwrite-strings -Wno-strict-aliasing -O3

# Options for the compile
OPTS = -Wall 

SYS_LIBS = -Wl,-rpath,/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib -Wl,-rpath,/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib -L/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib -lpetsc -lX11 -Wl,-rpath,/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib -L/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib -lHYPRE -lmpi_cxx -lstdc++ -lhdf5_fortran -lhdf5 -lz -lflapack -lfblas -L/curc/tools/free/redhat_5_x86_64/openmpi-1.4.3_gcc-4.5.2_torque-2.5.8_ib/lib -L/curc/tools/free/redhat_5_x86_64/gcc-4.5.2/lib64 -L/curc/tools/free/redhat_5_x86_64/gcc-4.5.2/lib/gcc/x86_64-unknown-linux-gnu/4.5.2 -L/curc/tools/free/redhat_5_x86_64/gcc-4.5.2/lib -L/curc/tools/nonfree/redhat_5_x86_64/ics_2011.0.013/composerxe-2011.0.084/compiler/lib/intel64 -L/curc/tools/nonfree/redhat_5_x86_64/ics_2011.0.013/composerxe-2011.0.084/mkl/lib/intel64 -ldl -lmpi -lopen-rte -lopen-pal -lnsl -lutil -lgcc_s -lpthread -lmpi_f90 -lmpi_f77 -lgfortran -lm -lm -lm -lm -lmpi_cxx -lstdc++ -ldl -lmpi -lopen-rte -lopen-pal -lnsl -lutil -lgcc_s -lpthread -ldl

PETSC_INCLUDE = -I/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/include -I/home/mmnasr/Petsc/petsc-3.1-p8/include -I/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/include -I/curc/tools/free/redhat_5_x86_64/openmpi-1.4.3_gcc-4.5.2_torque-2.5.8_ib/include -I/curc/tools/free/redhat_5_x86_64/openmpi-1.4.3_gcc-4.5.2_torque-2.5.8_ib/lib -I/home/mmnasr/Petsc/petsc-3.1-p8/linux-gnu-hdf5-o3/lib

INCLUDE = -I$(SRC_INCLUDE) -I$(PETSC_INCLUDE)

LIBS = $(SYS_LIBS)

OBJS= Velocity.o Memory.o Grid.o Solver.o MyMath.o Conc.o Pressure.o Input.o ENO.o Inflow.o Outflow.o Writer.o Output.o Display.o Debugger.o Resume.o Communication.o Surface.o Immersed.o Levelset.o Extract.o

gvg: $(OBJS)
	cd $(SRC_DIR); $(CC) -o $@ $@.c $(OPTS) $(OBJS) $(INCLUDE) $(PETSC_INC) $(LIBS); mv gvg $(TOP)

Pressure.o: $(SRC_DIR)/Pressure.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Conc.o: $(SRC_DIR)/Conc.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Input.o: $(SRC_DIR)/Input.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Grid.o: $(SRC_DIR)/Grid.c 
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Velocity.o: $(SRC_DIR)/Velocity.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Solver.o: $(SRC_DIR)/Solver.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Output.o: $(SRC_DIR)/Output.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Outflow.o: $(SRC_DIR)/Outflow.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Inflow.o: $(SRC_DIR)/Inflow.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Display.o: $(SRC_DIR)/Display.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)
	
ENO.o: $(SRC_DIR)/ENO.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

MyMath.o: $(SRC_DIR)/MyMath.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Memory.o: $(SRC_DIR)/Memory.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Writer.o: $(SRC_DIR)/Writer.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Debugger.o: $(SRC_DIR)/Debugger.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Resume.o: $(SRC_DIR)/Resume.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Communication.o: $(SRC_DIR)/Communication.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Levelset.o: $(SRC_DIR)/Levelset.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Surface.o: $(SRC_DIR)/Surface.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Immersed.o: $(SRC_DIR)/Immersed.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

Extract.o: $(SRC_DIR)/Extract.c
	cd $(SRC_DIR); $(CC) -c $*.c $(INCLUDE) $(OPTS)

clean:
	rm gvg *.bin *.dat *.info; cd $(SRC_DIR); rm *.o

run: 	
	$(MPIEXEC) -np $(THREADS) ./gvg
