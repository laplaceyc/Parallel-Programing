#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define true 1
#define false 0
#define MASTER_RANK 0
#define TAIL_SEND 0
#define HEAD_SEND 1

void swap(int* a, int* b){
	int tmp = *a;
		*a  = *b;
		*b  = tmp;
		return;
}

int HEADCOMPARE(int rank, int* local_buf, MPI_Comm comm, int sorted){
	int recv;
	//printf("function HEADCOMPARE on!My rank is%d\n",rank);
	MPI_Sendrecv(&local_buf[0], 1, MPI_INT, rank - 1, HEAD_SEND, &recv, 1, MPI_INT, rank - 1, TAIL_SEND, comm, MPI_STATUS_IGNORE);
			if (recv > local_buf[0]) {
				local_buf[0] = recv;
				sorted = false;
			}
	return sorted;
}
/*
				int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
				int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status)
*/

int TAILCOMPARE(int rank, int num_per_rank, int* local_buf, MPI_Comm comm, int sorted){
	int recvl;
	//printf("function TAILCOMPARE on!My rank is%d\n",rank);
	MPI_Sendrecv(&local_buf[num_per_rank-1], 1, MPI_INT, rank + 1, TAIL_SEND, &recvl, 1, MPI_INT, rank + 1, HEAD_SEND, comm, MPI_STATUS_IGNORE);
			if (recvl < local_buf[num_per_rank-1]) {
				local_buf[num_per_rank-1] = recvl;
				sorted = false;
			}
	return sorted;
}

int main (int argc, char *argv[]) {
	
	double communicate_time = 0, communicate_time_all = 0;
		
	int rank, size;
	int rc;
	MPI_File fh_in, fh_out;
	MPI_Offset offset;
	MPI_Comm custom_world = MPI_COMM_WORLD;
	MPI_Group origin_group, new_group;
	
	MPI_Init(&argc, &argv);
	double start_time = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	const int N = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];
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
	/*
	printf("Number of testcase is %d\n", N);
	printf("Rank is %d\n", rank);
	printf("size is %d\n", size); */
	// Read file using MPI-IO
	int num_per_rank = N / size;
	int remainder = N % size;
	// Two judge method
	
	if ((remainder) && (rank < (remainder))) {//allocate remainder N
		 num_per_rank++;
	}
	
	
	int head;
	
	if ((remainder)==0){
		head = rank * num_per_rank;
		//printf("I am in mode 1! \n");
	} else if(rank < (remainder)){
		head = rank * num_per_rank;
		//printf("I am in mode 2! \n");
	} else {
		head = (rank-(remainder))*num_per_rank + (remainder)*(num_per_rank+1);
		//printf("I am in mode 3! \n");
	}
	
	int tail = head + num_per_rank - 1;
	/*
	if (rank == (size - 1)) {//the last rank
		 num_per_rank += (N % size);
	}
	*/
	int *local_buf = new int[num_per_rank];
	
	if ((remainder)==0){
		offset = head * sizeof(int);
		//printf("I am in mode 1! \n");
	} else if(rank < (remainder)){
		offset = head * sizeof(int);
		//printf("I am in mode 2! \n");
	} else {
		offset = head* sizeof(int);
		//printf("I am in mode 3! \n");
	}
	
	rc = MPI_File_open(custom_world, INPUT_NAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);
	/*
	if (rc!=MPI_SUCCESS) {
		printf("Error File Open!!\n");
		MPI_Abort(custom_world,rc);
	}
	*/
	double IO_start_time_1 = MPI_Wtime();
	MPI_File_read_at(fh_in, offset, local_buf, num_per_rank, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh_in);
	double IO_end_time_1 = MPI_Wtime();
	/*printf("[Rank %d] num_per_node_size = %d\n" ,rank, num_per_rank);*/
	//printf("[Rank %d] \n" ,rank);
	/*
	for (int i = 0; i < num_per_rank; i++) printf("rank=%d local[%d]=%d",rank , i,local_buf[i]); fflush(stdout); 
	printf("\n");
	*/
	
	//printf("[END] \n" );
	// Odd-even sort
	int sorted = false, all_sorted = false;
	while (!all_sorted) {
		
		sorted = true;
		int i;
	//odd phase //local part
	if (head % 2 ==0){
		for(i = 0;i < num_per_rank;i += 2){
			if(i == 0) { continue; }
			if (local_buf[i] < local_buf[i-1]) {
				swap(&local_buf[i], &local_buf[i-1]);
				sorted = false;
			}
		}
	} else {
		for(i = 1;i < num_per_rank;i += 2){
				if (local_buf[i] < local_buf[i-1]) {
					swap(&local_buf[i], &local_buf[i-1]);
					sorted = false;
				}
		}
	}
	//odd phase //processor part
	double commu_start_time_1 = MPI_Wtime();
	if(rank != 0 && head % 2 == 0){
		sorted = HEADCOMPARE(rank, local_buf, custom_world, sorted);
		}
	
	if(rank != (size - 1) && tail % 2 == 1){
		sorted = TAILCOMPARE(rank, num_per_rank, local_buf, custom_world, sorted);
	}
	double commu_end_time_1 = MPI_Wtime();
	//even phase //local part
	if (head % 2 == 0){
		for(i = 1;i < num_per_rank;i += 2){
				if (local_buf[i] < local_buf[i-1]) {
					swap(&local_buf[i], &local_buf[i-1]);
					sorted = false;
				}
		}
	} else {
		for(i = 0;i < num_per_rank;i += 2){
			if(i == 0) { continue; }
				if (local_buf[i] < local_buf[i-1]) {
					swap(&local_buf[i], &local_buf[i-1]);
					sorted = false;
				}
		}		
	}
	//even phase //processor part
	double commu_start_time_2 = MPI_Wtime();
	if(rank != 0 && head % 2 == 1){
		sorted = HEADCOMPARE(rank, local_buf, custom_world, sorted);
		}
	
	if(rank != (size - 1) && tail % 2 == 0){
		sorted = TAILCOMPARE(rank, num_per_rank, local_buf, custom_world, sorted);
	}
	double commu_end_time_2 = MPI_Wtime();
	double sum1 = commu_end_time_1 - commu_start_time_1;
	double sum2 = commu_end_time_2 - commu_start_time_2;
	
	MPI_Barrier(custom_world);
	
	//printf("initial communicate_time = %f\n",communicate_time);
	communicate_time += (sum1 + sum2);
	//printf("finial communicate_time = %f\n",communicate_time);
	MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_LAND, custom_world);
	}
	MPI_Barrier(custom_world);
	MPI_Allreduce(&communicate_time, &communicate_time_all, 1, MPI_DOUBLE, MPI_SUM, custom_world);
	
	

	//printf("Now for show the result[Rank %d] \n" ,rank);
	/*
	for (int i = 0; i < num_per_rank; i++) printf("rank=%d n=%d",rank ,local_buf[i]);
        printf("\n");
	*/
	// Write file using MPI-IO
	double IO_start_time_2 = MPI_Wtime();
	MPI_File_open(custom_world, OUTPUT_NAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);
	MPI_File_write_at(fh_out, offset, local_buf, num_per_rank, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh_out);
	double IO_end_time_2 = MPI_Wtime();
	/*	
        for (int i = 0; i < N; ++i)
             printf("[START] [Rank %d] local_arr[%d] = %d\n", rank, i, local_buf[i]);
	*/
	delete [] local_buf;
	

	MPI_Barrier(custom_world);
	double end_time = MPI_Wtime();
	double sum3 = IO_end_time_1-IO_start_time_1;
	double sum4 = IO_end_time_2-IO_start_time_2;
	
	if(rank==0){
	printf("------------------RESULT---------------------\n");
	printf("total exec. time = %f\n", end_time - start_time);
	printf("I/O readtime = %f, I/O writetime = %f\n",sum3, sum4);
	printf("total I/O time = %f\n", sum3 + sum4);
	printf("communicate time = %f\n",communicate_time_all);
	}
		MPI_Finalize();
	return 0;}
