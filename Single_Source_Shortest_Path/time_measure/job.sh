#PBS -N SSSP_Test
#PBS -r n
#PBS -l nodes=1:ppn=5
#PBS -l walltime=00:05:00

export MV2_ENABLE_AFFINITY=0	# prevent MPI from binding all threads to one core

cd $PBS_O_WORKDIR
mpiexec -np 5 ./SSSP_MPI_async 10 In_5_4 out1 3

