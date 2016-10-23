#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define true 1
#define false 0
#define TAIL_SEND 0
#define HEAD_SEND 1
//swap function
void swap(int* a, int* b){
	int tmp = *a;
		*a  = *b;
		*b  = tmp;
		return;
}
//compare my head with the previous rank tail 
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
//compare my tail with the next rank head 
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
	int rank, size;
	int rc;
	MPI_File fh_in, fh_out;
	MPI_Offset offset;
	MPI_Comm custom_world = MPI_COMM_WORLD;
	MPI_Group origin_group, new_group;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	const int N = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];
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
	int num_per_rank = N / size;
	int head = rank * num_per_rank;//index number
	int tail = head + num_per_rank - 1;//index number
	offset = rank * num_per_rank * sizeof(int);// assign un-allocate remainder offset(It is very important for the last processor)
	
	if (rank == (size - 1)) {//the last rank assigned the remainder (N % size)
		 num_per_rank += (N % size);
	}
	int *local_buf = new int[num_per_rank+1];
	
	// Read file using MPI-IO
	rc = MPI_File_open(custom_world, INPUT_NAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);
	// Detection file open state
	if (rc!=MPI_SUCCESS) {
		printf("File open failed!!\n");
		MPI_Abort(custom_world,rc);
	}
	MPI_File_read_at(fh_in, offset, local_buf, num_per_rank, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh_in);
	
	// Start Odd-even sort
	int sorted = false, all_sorted = false;
	while (!all_sorted) {
		sorted = true;
		int i;
	//odd phase //local-sort part
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
	//odd phase //processor-communication-sort part
	if(rank != 0 && head % 2 == 0){
		sorted = HEADCOMPARE(rank, local_buf, custom_world, sorted);
	}
	if(rank != (size - 1) && tail % 2 == 1){
		sorted = TAILCOMPARE(rank, num_per_rank, local_buf, custom_world, sorted);
	}
	//even phase //local-sort part
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
	//even phase ////processor-communication-sort part
	if(rank != 0 && head % 2 == 1){
		sorted = HEADCOMPARE(rank, local_buf, custom_world, sorted);
	}
	if(rank != (size - 1) && tail % 2 == 0){
		sorted = TAILCOMPARE(rank, num_per_rank, local_buf, custom_world, sorted);
	}
	//wait until all processor sort complete
	MPI_Barrier(custom_world);
	MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_LAND, custom_world);
	}
	
	// Write file using MPI-IO
	MPI_File_open(custom_world, OUTPUT_NAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);
	MPI_File_write_at(fh_out, offset, local_buf, num_per_rank, MPI_INT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh_out);
	
	//free unused buffer
	delete [] local_buf;

	MPI_Barrier(custom_world);
	MPI_Finalize();

	return 0;}
