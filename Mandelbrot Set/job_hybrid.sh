
# NOTICE: Please do not remove the '#' before 'PBS'

# Name of your job
#PBS -N HYBRID

# Declaring job as not re-runnable
#PBS -r n

# Resource allocation (how many nodes? how many processes per node?)
#PBS -l nodes=3:ppn=12

# Max execution time of your job (hh:mm:ss)
# Your job may got killed if you exceed this limit
#PBS -l walltime=00:05:00

cd $PBS_O_WORKDIR
NUM_MPI_PROCESS_PER_NODE=1	# edit this line to set number of MPI process you want to use per node
export OMP_NUM_THREADS=12	# set max number of threads OpenMP can use per MPI task
export MV2_ENABLE_AFFINITY=0	# prevent MPI from binding all threads to one core
# Please make sure that NUM_MPI_PROCESS_PER_NODE * OMP_NUM_THREADS == ppn

time mpiexec -ppn $NUM_MPI_PROCESS_PER_NODE ./MS_Hybrid_static $OMP_NUM_THREADS -2 2 -2 2 600 600 disable  # edit this line to fit your needs!
# In this case, it will use 3 nodes to run, each nodes with 3 processes, each process with 4 threads. (3 process * 4 thread = 12 ppn )
# Besides NUM_MPI_PROCESS_PER_NODE, OMP_NUM_THREADS, you also need to modify the number of nodes or ppn you want to use.
