CC			:= gcc
CXX			:= g++

MPICC			:= mpicc
MPICXX			:= mpicxx

CCFLAGS			:= -O3 -march=native -Wall -std=gnu11
CXXFLAGS		:= -O3 -march=native -Wall -std=c++0x

PFLAGS			:= -lpthread

all: SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async

SSSP_Pthread: SSSP_Pthread.cpp
	$(CXX) $(CXXFLAGS) $(PFLAGS) -o $@ $?
SSSP_MPI_sync: SSSP_MPI_sync.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $?
SSSP_MPI_async: SSSP_MPI_async.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $?

clean:
	rm -f SSSP_Pthread SSSP_MPI_sync SSSP_MPI_async
