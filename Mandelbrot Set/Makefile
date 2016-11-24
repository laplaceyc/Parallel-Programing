CC			:= gcc
CXX			:= g++

MPICC			:= mpicc
MPICXX			:= mpic++

CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
CXXFLAGS		:= -O3 -march=native -Wall -std=c++0x
LDFLAGS			:= -lX11 -fopenmp -lm 

all: MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic MS_seq

MS_MPI_static: MS_MPI_static.cpp
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_MPI_dynamic: MS_MPI_dynamic.cpp
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_OpenMP_static: MS_OpenMP_static.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_OpenMP_dynamic: MS_OpenMP_dynamic.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_Hybrid_static: MS_Hybrid_static.cpp
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_Hybrid_dynamic: MS_Hybrid_dynamic.cpp
	$(MPICXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
MS_seq: MS_sequential.c
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $?
clean:
	rm -f MS_MPI_static MS_MPI_dynamic MS_OpenMP_static MS_OpenMP_dynamic MS_Hybrid_static MS_Hybrid_dynamic MS_seq
