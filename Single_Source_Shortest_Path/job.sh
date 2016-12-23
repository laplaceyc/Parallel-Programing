#PBS -N Pthread_64
#PBS -r n
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:05:00

export MV2_ENABLE_AFFINITY=0	# prevent MPI from binding all threads to one core

cd $PBS_O_WORKDIR
./SSSP_Pthread 1 In_64_1024 out1 3

