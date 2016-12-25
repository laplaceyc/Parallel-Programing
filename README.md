# Parallel-Programing
This repo is "NTHU Parallel Programing" course project.

# HW1 Odd-Even Sort
Use MPI to do "Odd-Even Sort".

There are two versions by the following restriction.
- Basic implementation

Element level odd-even sort, each element can only be swapped with its adjacent elements in each operation.
- Advances implementation

Process level odd-even sort, only the communication pattern between processes is restricted.

# HW2 Mandelbrot Set
Use MPI and OpenMP to do "Mandelbrot Set".

There are three versions by using

- Distributed memory - MPI
- Shared memory - OpenMP
- Hybrid (distributed-shared) memory - MPI + OpenMP

And for three different memory architecture memtioned above (MPI/OpenMP/Hybrid), each of them also has two different scheduling policies

- Static scheduling
- Dynamic scheduling

# HW3 Single Source Shortest Path
Use Pthread and MPI to do "Single Source Shortest Path".

There are three version by using

- Pthread: without amy limitation or constraints
- Fully distributed synchronous vertex-centric MPI
- Fully distributed asynchronous vertex-centric MPI

The specification of two MPI versions can observe "PP_2016_HW3_v2" in folder and will explain in detail.
