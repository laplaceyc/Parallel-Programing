#PBS -N Async_256_04_01
#PBS -r n
#PBS -l nodes=4:ppn=1
#PBS -l walltime=00:00:30

export MV2_ENABLE_AFFINITY=0	# prevent MPI from binding all threads to one core

cd $PBS_O_WORKDIR
mpiexec -np 256 ./MPI_async_time 4 In_256_2048 Out_256_2048 30
