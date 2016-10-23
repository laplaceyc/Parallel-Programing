#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <algorithm>
#define true 1
#define false 0
#define LEFT_SEND 0
#define RIGHT_SEND 1
/*
void LEFTCOMPARE(int rank,int num_per_rank,int num_per_rank_Left, int* local_buf, MPI_Comm comm){
	int recv_buf[num_per_rank_Left+1];
	int out[num_per_rank+num_per_rank_Left+1];
	int i,j;
	int outcount=0;
	MPI_Sendrecv(&local_buf, num_per_rank, MPI_INT, rank-1, LEFT_SEND, recv_buf, num_per_rank_Left, MPI_INT,  rank-1, RIGHT_SEND, comm, MPI_STATUS_IGNORE);
	//merge
	for (i=0,j=0; i<num_per_rank; i++) {
        while ((recv_buf[j] < local_buf[i]) && j < num_per_rank_Left) {
            out[outcount++] = recv_buf[j++];
        }
        out[outcount++] = local_buf[i];
    }
    while (j<num_per_rank_Left) out[outcount++] = recv_buf[j++];
	//allocate
	for (int i=num_per_rank_Left; i < num_per_rank + num_per_rank_Left; i++) local_buf[i] = out[i];
	return;
}
void RIGHTCOMPARE(int rank,int num_per_rank,int num_per_rank_Right, int* local_buf, MPI_Comm comm){
	int recv_buf[num_per_rank_Right+1];
	int out[num_per_rank+num_per_rank_Right+1];
	int i,j;
	int outcount=0;
	MPI_Sendrecv(&local_buf, num_per_rank, MPI_INT, rank+1, RIGHT_SEND, recv_buf, num_per_rank_Right, \
					MPI_INT,  rank+1, LEFT_SEND, comm, MPI_STATUS_IGNORE);
	//merge
	for (i=0,j=0; i<num_per_rank; i++) {
        while ((recv_buf[j] < local_buf[i]) && j < num_per_rank_Right) {
            out[outcount++] = recv_buf[j++];
        }
        out[outcount++] = local_buf[i];
    } 
    while (j<num_per_rank_Right) out[outcount++] = recv_buf[j++];
	//allocate take small
	for (int i=0; i < num_per_rank; i++) local_buf[i] = out[i];
	return;	
}*/
/*
				int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
				int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status)
*/

int main (int argc, char *argv[]) {
	int rank, size;
	//int rc;
	MPI_File fh_in, fh_out;
	MPI_Offset offset;
	MPI_Status status;
	MPI_Comm custom_world = MPI_COMM_WORLD;
	MPI_Group origin_group, new_group;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	const int N = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];
	int remainder = N % size;
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
	
	// Read file using MPI-IO
	int num_per_rank = N / size;
	// Two judge method
	if ((remainder) && (rank < (remainder))) {//allocate remainder N
		 num_per_rank++;
	}
	
	/*
	if (rank == (size - 1)) {//the last ranl
		 num_per_rank += (N % size);
	}
	*/
	int *local_buf = new int[num_per_rank];
	if ((remainder)==0){
		offset = rank * num_per_rank * sizeof(int);
		//printf("I am in mode 1! \n");
	} else if(rank < (remainder)){
		offset = rank * num_per_rank * sizeof(int);
		//printf("I am in mode 2! \n");
	} else {
		offset = ((rank-(remainder))*num_per_rank + (remainder)*(num_per_rank+1))* sizeof(int);
		//printf("I am in mode 3! \n");
	}

	MPI_File_open(custom_world, INPUT_NAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh_in);
	/*if (rc!=MPI_SUCCESS) {
	printf("Error File Open!!\n");
	MPI_Abort(MPI_COMM_WORLD,rc);
	}*/
	MPI_File_read_at(fh_in, offset, local_buf, num_per_rank, MPI_INT, &status);
	MPI_File_close(&fh_in);
	printf("[Rank %d] num_per_node_size = %d\n" ,rank, num_per_rank); 
	//for (int i = 0; i < num_per_rank; i++)
    //    	printf("[START] [Rank %d] local_buf[%d] =%d\n", rank, i, local_buf[i]); fflush(stdin);
	std::sort(local_buf, local_buf + num_per_rank);//local sort
	//determine dest rank num
	int num_per_rank_Left, num_per_rank_Right;
	if (remainder == 0) num_per_rank_Left = num_per_rank;
	else num_per_rank_Left = (rank == remainder) ? num_per_rank + 1 : num_per_rank;
	
	if (remainder == 0) num_per_rank_Right = num_per_rank;
	else num_per_rank_Right = (rank == remainder - 1) ? num_per_rank - 1 : num_per_rank;
	
	int *recv_bufl = new int[num_per_rank_Left];
	int *recv_bufr = new int[num_per_rank_Right];
	int *out = new int[num_per_rank*2+1];
	//then merge
	for (int i = 1; i <= size; i++) {//processor level odd-even sort
	//odd phase sort
		if((rank != 0)&&((rank + i) % 2==0)) {	
		int i,j;
		int outcount=0;
		MPI_Sendrecv(&local_buf, num_per_rank, MPI_INT, rank-1, LEFT_SEND, recv_bufl, num_per_rank_Left, MPI_INT,  rank-1, RIGHT_SEND, custom_world, MPI_STATUS_IGNORE);
		for (i=0,j=0; i < num_per_rank; i++) {
        while ((recv_bufl[j] < local_buf[i]) && j < num_per_rank_Left) {
            out[outcount++] = recv_bufl[j++];
        }
        out[outcount++] = local_buf[i];
		}
		while (j<num_per_rank_Left) out[outcount++] = recv_bufl[j++];
		for (int i=num_per_rank_Left; i < num_per_rank + num_per_rank_Left; i++) local_buf[i] = out[i];
		}
		
		else if (rank != size-1){
		int i,j;
		int outcount=0;
		MPI_Sendrecv(&local_buf, num_per_rank, MPI_INT, rank+1, RIGHT_SEND, recv_bufr, num_per_rank_Right, \
					 MPI_INT,  rank+1, LEFT_SEND, custom_world, MPI_STATUS_IGNORE);
		for (i=0,j=0; i<num_per_rank; i++) {
			while ((recv_bufr[j] < local_buf[i]) && j < num_per_rank_Right) {
            out[outcount++] = recv_bufr[j++];
        }
        out[outcount++] = local_buf[i];
		} 
		while (j<num_per_rank_Right) out[outcount++] = recv_bufr[j++];
	//allocate take small
		for (int i=0; i < num_per_rank; i++) local_buf[i] = out[i];
		}
    }
	delete [] recv_bufl;
	delete [] recv_bufr;
	delete [] out;
	
	// Write file using MPI-IO
	MPI_File_open(custom_world, OUTPUT_NAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh_out);
	MPI_File_write_at(fh_out, offset, local_buf, num_per_rank, MPI_INT, &status);
	MPI_File_close(&fh_out);
	/*	
        for (int i = 0; i < N; ++i)
             printf("[START] [Rank %d] local_arr[%d] = %d\n", rank, i, local_buf[i]);
	*/
	delete [] local_buf;

	

	MPI_Barrier(custom_world);
	MPI_Finalize();

	return 0;}
