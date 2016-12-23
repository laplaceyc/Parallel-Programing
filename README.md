# Parallel-Programing
This repo is "NTHU Parallel Programing" course project.

# HW1 Odd-Even Sort
Use MPI to do "Odd-Even Sort".<br>
There are two versions by the following restriction.<br>
&nbsp;&nbsp;●Basic implementation<br>
&nbsp;&nbsp;Element level odd-even sort, each element can only be swapped with its adjacent elements in each operation.<br>
&nbsp;&nbsp;●Advances implementation<br>
&nbsp;&nbsp;Process level odd-even sort, only the communication pattern between processes is restricted.<br>
# HW2 Mandelbrot Set
Use MPI and OpenMP to do "Mandelbrot Set".<br>
There are three versions by using<br>
&nbsp;&nbsp;●Distributed memory - MPI<br>
&nbsp;&nbsp;●Shared memory - OpenMP<br>
&nbsp;&nbsp;●Hybrid (distributed-shared) memory - MPI + OpenMP<br>
And for three different memory architecture memtioned above (MPI/OpenMP/Hybrid), each of them also has two different scheduling policies<br>
&nbsp;&nbsp;●Static scheduling<br>
&nbsp;&nbsp;●Dynamic scheduling<br>
# HW3 Single Source Shortest Path
Use Pthread and MPI to do "Single Source Shortest Path".<br>
There are three version by using<br>
&nbsp;&nbsp;●Pthread: without amy limitation or constraints<br>
&nbsp;&nbsp;●Fully distributed synchronous vertex-centric MPI<br>
&nbsp;&nbsp;●Fully distributed asynchronous vertex-centric MPI<br>
The specification of two MPI versions can observe "PP_2016_HW3_v2" in folder and will explain in detail.
