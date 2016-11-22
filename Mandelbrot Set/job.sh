
# NOTICE: Please do not remove the '#' before 'PBS'

# Select interactive queue for Xwindow
#PBS -q interactive

# Name of your job
#PBS -N MY_JOB

# X11 Forwarding
#PBS -X -I

# Resource allocation (how many nodes? how many processes per node?)
# For interactive queue, the only resources you can request is "nodes=1:ppn=12"
#PBS -l nodes=1:ppn=12

# Max execution time of your job (hh:mm:ss)
# Your job may got killed if you exceed this limit
# For interactive queue, the maximum execution time you can request is 30 min
#PBS -l walltime=00:30:00

# Do not add anything below
