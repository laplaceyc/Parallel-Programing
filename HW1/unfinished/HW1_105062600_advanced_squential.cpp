#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <algorithm>
#define LEFT_SEND 0
#define RIGHT_SEND 1
//merge sort function
void merge(int *local_buf, int num_per_rank, int *recv_buf, int recv_num_per_rank, int *out) {
    int i,j;
    int outcount=0;

    for (i=0,j=0; i < num_per_rank; i++) {
        while ((recv_buf[j] < local_buf[i]) && j < recv_num_per_rank) {
            out[outcount++] = recv_buf[j++];
        }
        out[outcount++] = local_buf[i];
    }
    while (j<recv_num_per_rank)
        out[outcount++] = recv_buf[j++];

    return;
}

int main (int argc, char *argv[]) {
	int rank, size;
	int rc;
	MPI_File fh_in, fh_out;
	MPI_Status status;
	MPI_Comm custom_world = MPI_COMM_WORLD;
	MPI_Group origin_group, new_group;

	MPI_Init(&argc, &argv);
	double start_time = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	const int N = atoi(argv[1]);
	int remainder = N % size;
	//Produces a group by excluding ranges of processes from an existing group
	if (N < size) {
		// obtain the group of proc. in the world communicator
		MPI_Comm_group(custom_world, &origin_group);
		// remove unwanted ranks
		int ranges[][3] = {{N, size-1, 1}};
		MPI_Group_range_excl(origin_group, 1, ranges, &new_group);
		// create a new communicator
		MPI_Comm_create(custom_world, new_group, &custom_world);
		if (custom_world == MPI_COMM_NULL) {
			// terminate those unwanted processes
			MPI_Finalize();
			exit(0);
		}
		size = N;
	}
	
	//Declare parameter
	const int LAST = size - 1;
	int num_per_rank = N / size;
	int fix_size = num_per_rank;
	if (rank == (size - 1)) {//the last rank assigned the remainder (N % size)
		 num_per_rank += (N % size);
	}
	int displ_offset[size];
	int displ_num[size];
	int *root_buf;
	
	for (int i =0;i < size; i++)
	{
		if(i==0)
		{
			displ_offset[i]=0;
		}
		else
		{
			displ_offset[i] = fix_size * i;
		}
		displ_num[i] = num_per_rank;		
	}
	int *local_buf = new int[num_per_rank];
	
	// Read file using squential-I/O
	FILE *INPUT_NAME ;
	double IO_start_time_1 = MPI_Wtime();;
	if(rank==0)	{
		root_buf = new int[N];
		INPUT_NAME = fopen(argv[2],"rb");
		for (int i = 0; i < N; i++)
		{	
			  fread(&root_buf[i],sizeof(int),1,INPUT_NAME);
		}
		
	}
	MPI_Scatterv(&root_buf,displ_num,displ_offset,MPI_INT,&local_buf,displ_num[rank],MPI_INT,0,custom_world);
	fclose(INPUT_NAME);
	double IO_end_time_1 = MPI_Wtime();
	//use C algorithm library to execute local sort
	std::sort(local_buf, local_buf + num_per_rank);
	
	
	//use merge sort between processor
	int* recv_buf = new int[fix_size];
	int* out = new int[(num_per_rank + fix_size)];
	double commu_start_time_1 = MPI_Wtime();
	for (int j = 1; j <= size; j++) {//processor level odd-even sort
	//odd and even phase sort j is odd for odd phase and even for even phase
		if((rank != LAST)&&((rank + j) % 2==0)){
			MPI_Send(local_buf, fix_size, MPI_INT, rank + 1, LEFT_SEND, custom_world);
			MPI_Recv(local_buf, fix_size, MPI_INT, rank + 1, RIGHT_SEND, custom_world, MPI_STATUS_IGNORE);
		}
		
		else if (((rank + j) % 2==1)&&(rank != 0)) {
			MPI_Recv(recv_buf, fix_size, MPI_INT, rank - 1, LEFT_SEND, custom_world, MPI_STATUS_IGNORE);
			merge(local_buf, num_per_rank, recv_buf, fix_size, out);
			for (int k = fix_size; k < fix_size + num_per_rank; k++) local_buf[k-fix_size] = out[k];
			MPI_Send(out, fix_size, MPI_INT, rank - 1, RIGHT_SEND, custom_world);
		}
		//wait until all processor sort complete
		MPI_Barrier(custom_world);
    }
	double commu_end_time_1 = MPI_Wtime();
	double sum1 = commu_end_time_1 - commu_start_time_1;
	//free unused buffer
	delete [] recv_buf;
	delete [] out;
	FILE *OUTPUT_NAME;
	
	// Write file using squen I/O
	double IO_start_time_2 = MPI_Wtime();
	MPI_Gatherv(&local_buf,displ_num[rank],MPI_INT,&root_buf,displ_num,displ_offset,MPI_INT,0,custom_world);
	if(rank==0)
	{
		OUTPUT_NAME = fopen(argv[3],"wb");
		for (int i = 0; i < N; i++)
		{	
			 fwrite(&root_buf[i],sizeof(int),1,OUTPUT_NAME);
		}
		
	}
	fclose(OUTPUT_NAME);
	double IO_end_time_2 = MPI_Wtime();
	//free unused buffer
	if (rank != 0) {
        delete []  local_buf;
     //   printf("[FREE] [RANK %d] SUCCESS FREE\n", rank);
    } else {
        delete [] root_buf;
        delete [] local_buf;
    }
	
	MPI_Barrier(custom_world);
	double end_time = MPI_Wtime();
	double sum3 = IO_end_time_1-IO_start_time_1;
	double sum4 = IO_end_time_2-IO_start_time_2;
	if(rank==0){
	printf("------------------RESULT---------------------\n");
	printf("total exec. time = %f\n", end_time - start_time);
	printf("I/O readtime = %f, I/O writetime = %f\n",sum3, sum4);
	printf("total I/O time = %f\n", sum3 + sum4);
	printf("communicate time = %f\n",sum1);
	}
	MPI_Finalize();

	return 0;}
