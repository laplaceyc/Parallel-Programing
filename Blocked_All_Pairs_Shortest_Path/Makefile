CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_20 -O3

OMPFLAGS                := -fopenmp
MPILIBS                 := -I/opt/mpich-3.2-gcc/include -L/opt/mpich-3.2-gcc/lib -lmpich

EXES                    := HW4_cuda.exe HW4_openmp.exe HW4_mpi.exe

all: $(EXES)

HW4_cuda.exe: HW4_105062600_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_openmp.exe: HW4_105062600_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

HW4_mpi.exe: HW4_105062600_mpi.cu
	$(NVCC) $(NVFLAGS) $(MPILIBS) -o $@ $?

clean:
	rm -f $(EXES)

