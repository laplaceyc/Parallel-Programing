#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

#define INF 10000000
//location the row major position on 1Darray
__device__ __host__ int location(int i, int j, int N) {
	return i*N+j;
}
//judge the number
__device__ int judge(int a, int b, int c){
	return (a > b + c) ? b + c : a;
}

int* host_matrix;
int* device_matrix;

__global__ void FW_phase1(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
    
	bi = r;
	bj = r;
    
	extern __shared__ int DS[];

	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	// DS[0:bibs-1][:] = B[bi][bj] = din[bibs:(bi+1)bs-1][bjbs:(bj+1)bs-1]
	// DS[bibs:2bibs-1][:] = B[bi][r] = din[bibs:(bi+1)bs-1][rbs:(r+1)bs-1]
	// DS[2bibs:3bibs-1][:] = B[r][bi] = din[rbs:(r+1)bs-1][bjbs:(bj+1)bs-1]
	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();

	// DS[i][j] = min{ DS[i][j], DS[i+bs][k] + DS[k+2bs][j] }
	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];
            DS[location(i+2*blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
            DS[location(i+blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
		}	
//		DS[location(i, j, blocking_factor)] = judge(DS[location(i, j, blocking_factor)], DS[location(i+blocking_factor, k, blocking_factor)] ,DS[location(k+2*blocking_factor, j, blocking_factor)]);
//	__syncthreads();
	}

	// DS[i][j] = din[i+bsbi][j+bsbj]
	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
	__syncthreads();
}

__global__ void FW_phase2(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM) {
	int bi, bj;
	
	if (blockIdx.x == 1) {
		//column
		bi = (r + blockIdx.y + 1) % (N/blocking_factor);
		bj = r;
	} else {
		//row
		bi = r;
		bj = (r + blockIdx.y + 1) % (N/blocking_factor);
            }

	extern __shared__ int DS[];
	
	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	// DS[0:bibs-1][:] = B[bi][bj] = din[bibs:(bi+1)bs-1][bjbs:(bj+1)bs-1]
	// DS[bibs:2bibs-1][:] = B[bi][r] = din[bibs:(bi+1)bs-1][rbs:(r+1)bs-1]
	// DS[2bibs:3bibs-1][:] = B[r][bi] = din[rbs:(r+1)bs-1][bjbs:(bj+1)bs-1]
	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();

	// DS[i][j] = min{ DS[i][j], DS[i+bs][k] + DS[k+2bs][j] }
	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];
            if (r == bi) DS[location(i+2*blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
            if (r == bj) DS[location(i+blocking_factor, j, blocking_factor)] = DS[location(i, j, blocking_factor)];
		}	
//		DS[location(i, j, blocking_factor)] = judge(DS[location(i, j, blocking_factor)], DS[location(i+blocking_factor, k, blocking_factor)] ,DS[location(k+2*blocking_factor, j, blocking_factor)]);
//    __syncthreads();
	}
	// DS[i][j] = din[i+bsbi][j+bsbj]
	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
	__syncthreads();
}
__global__ void FW_phase3(int* din, int blocking_factor, int N, int r, int MAX_BLCOK_DIM, int offset) {
	int bi, bj;
    
	bi = blockIdx.x + offset;
	bj = blockIdx.y;
     
	extern __shared__ int DS[];
	
	int offset_i = blocking_factor * bi;
	int offset_j = blocking_factor * bj;
	int offset_r = blocking_factor * r;

	int i = threadIdx.y;
	int j = threadIdx.x;

	// DS[0:bibs-1][:] = B[bi][bj] = din[bibs:(bi+1)bs-1][bjbs:(bj+1)bs-1]
	// DS[bibs:2bibs-1][:] = B[bi][r] = din[bibs:(bi+1)bs-1][rbs:(r+1)bs-1]
	// DS[2bibs:3bibs-1][:] = B[r][bi] = din[rbs:(r+1)bs-1][bjbs:(bj+1)bs-1]
	DS[location(i, j, blocking_factor)] = din[location(i+offset_i, j+offset_j, N)];
	DS[location(i+blocking_factor, j, blocking_factor)] = din[location(i+offset_i, j+offset_r, N)];
	DS[location(i+2*blocking_factor, j, blocking_factor)] = din[location(i+offset_r, j+offset_j, N)];
	__syncthreads();
	
    // DS[i][j] = min{ DS[i][j], DS[i+bs][k] + DS[k+2bs][j] }
	for (int k = 0; k < blocking_factor; k++) {
		if (DS[location(i, j, blocking_factor)] > DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)]) {
            DS[location(i, j, blocking_factor)] = DS[location(i+blocking_factor, k, blocking_factor)] + DS[location(k+2*blocking_factor, j, blocking_factor)];

		}	
//		DS[location(i, j, blocking_factor)] = judge(DS[location(i, j, blocking_factor)], DS[location(i+blocking_factor, k, blocking_factor)] ,DS[location(k+2*blocking_factor, j, blocking_factor)]);
//	__syncthreads();
	}
	
	// DS[i][j] = din[i+bsbi][j+bsbj]
	din[location(i+offset_i, j+offset_j, N)] = DS[location(i, j, blocking_factor)];
	__syncthreads();
}
int main(int argc, char* argv[]) {
/*	
	//time measurement
	float elapsed_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
*/	
	float IO_time = 0.0;
	float memory_time = 0.0;
	float comp_time = 0.0;
	float comm_time = 0.0;
	float t,t_m,t_e,t_c;
	cudaEvent_t start, stop;
	cudaEvent_t mem_s, mem_e;
	cudaEvent_t comp_s, comp_e;
	cudaEvent_t comm_s, comm_e;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&mem_s);
	cudaEventCreate(&mem_e);
	cudaEventCreate(&comp_s);
	cudaEventCreate(&comp_e);
	cudaEventCreate(&comm_s);
	cudaEventCreate(&comm_e);
	if (argc != 4) {
		fprintf(stderr, "Insuficcient arguments\n");
		fprintf(stderr, "Usage: ./%s input_file output_file blocking_factor\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	//load argument
	const char *INPUT_NAME = argv[1];
	const char *OUTPUT_NAME = argv[2];
	int blocking_factor = atoi(argv[3]);

	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	cudaSetDevice(rank);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, rank);
	printf("Dev%d. Name: %s\n", rank, prop.name);


	// read file
	//I/O
	cudaEventRecord(start);
	FILE *fh_in, *fh_out;
	fh_in = fopen(INPUT_NAME, "r");
	int edge_num, vertex_num;
	fscanf(fh_in, "%d %d", &vertex_num, &edge_num);
	//I/O
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	IO_time += t;
	if (blocking_factor > vertex_num) blocking_factor = vertex_num;
	int VERTEX_EXT = vertex_num + (blocking_factor - ((vertex_num-1) % blocking_factor + 1));
 
	if(rank == 0) printf("Blocking factor: %d\n", blocking_factor);

	//allocate memory
	cudaMallocHost((void**) &host_matrix, sizeof(int) * VERTEX_EXT*VERTEX_EXT);
	cudaMalloc((void**) &device_matrix, sizeof(int) * VERTEX_EXT*VERTEX_EXT);


	//initialize (store data in row major)
	for(int i = 0; i < VERTEX_EXT; i++){
		for(int j = 0; j < VERTEX_EXT; j++){
			if(i == j) host_matrix[i * VERTEX_EXT + j] = 0;
			else host_matrix[i * VERTEX_EXT + j] = INF;
		}
	}
	//I/O
	cudaEventRecord(start);
	int a, b, weight;//a and b is source_vertex and destination_vertex respectively 
	for(int i = 0; i < edge_num; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		a -= 1;
		b -= 1;
		host_matrix[a * VERTEX_EXT + b] = weight;
	}
	//I/O
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	IO_time += t;
	fclose(fh_in);
//	printf("[%d]read file finish\n",rank);
	//start FW algorithm
	//Determine block dimension
	int MAX_BLCOK_DIM = blocking_factor > 32 ? 32 : blocking_factor;
	//Declare 3D parameter
	dim3 BLOCK_DIM(MAX_BLCOK_DIM, MAX_BLCOK_DIM);
	
	
	int blocks[3];
	//phase 1
	dim3 grid_phase1(1);
	// phase 2
	int round =  (VERTEX_EXT + blocking_factor - 1) / blocking_factor;
	blocks[1] = round;
	dim3 grid_phase2(2, blocks[1]-1);
	// phase 3
	int num_blocks_per_thread = round / size;
	int row_offset = num_blocks_per_thread * rank * blocking_factor;
	if (rank == size - 1) num_blocks_per_thread += round % size;
	dim3 grid_phase3(num_blocks_per_thread, round);
	
	int cpy_idx = location(row_offset, 0, VERTEX_EXT);
//	printf("round = %d",round);
	if(rank == 0) cudaEventRecord(mem_s);
	cudaMemcpy((void*) &(device_matrix[cpy_idx]), (void*) &(host_matrix[cpy_idx]), sizeof(int) * VERTEX_EXT*blocking_factor*num_blocks_per_thread, cudaMemcpyHostToDevice);	
	if(rank == 0){
		cudaEventRecord(mem_e);
		cudaEventSynchronize(mem_e);
		cudaEventElapsedTime(&t_m, mem_s, mem_e);
		memory_time += t_m;
		}
//	printf("[%d]allcoate finish. Start FW\n",rank);		
	for (int r = 0; r < round; r++) {
//		printf("[%d]in cycle round =%d\n",rank,r);		
		int r_idx = location(r * blocking_factor, 0, VERTEX_EXT);

		if (r >= row_offset/blocking_factor && r < (row_offset/blocking_factor + num_blocks_per_thread)) {
			if(rank == 0) cudaEventRecord(mem_s);
            cudaMemcpy((void*) &(host_matrix[r_idx]), (void*) &(device_matrix[r_idx]), sizeof(int) * VERTEX_EXT * blocking_factor, cudaMemcpyDeviceToHost);
			if(rank == 0){
			cudaEventRecord(mem_e);
			cudaEventSynchronize(mem_e);
			cudaEventElapsedTime(&t_m, mem_s, mem_e);
			memory_time += t_m;
			}	
			if(rank == 0) cudaEventRecord(comm_s);
            // MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
            MPI_Send(&host_matrix[r_idx], VERTEX_EXT * blocking_factor, MPI_INT, (rank + 1) % 2, 0, MPI_COMM_WORLD);
			if(rank == 0){
			cudaEventRecord(comm_e);
			cudaEventSynchronize(comm_e);
			cudaEventElapsedTime(&t_c, comm_s, comm_e);
			comm_time += t_c; 
	  }
        } else {
			if(rank == 0) cudaEventRecord(comm_s);
            // MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
            MPI_Recv(&host_matrix[r_idx], VERTEX_EXT * blocking_factor, MPI_INT, (rank + 1) % 2, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(rank == 0){
			cudaEventRecord(comm_e);
			cudaEventSynchronize(comm_e);
			cudaEventElapsedTime(&t_c, comm_s, comm_e);
			comm_time += t_c; 
			}
        }
		
//		printf("[%d]End of data transfer\n",rank);	
		if(rank == 0) cudaEventRecord(mem_s);
		cudaMemcpy((void*) &(device_matrix[r_idx]), (void*) &(host_matrix[r_idx]), sizeof(int) * VERTEX_EXT * blocking_factor, cudaMemcpyHostToDevice);
		if(rank == 0){
		cudaEventRecord(mem_e);
		cudaEventSynchronize(mem_e);
		cudaEventElapsedTime(&t_m, mem_s, mem_e);
		memory_time += t_m;
		}
		if(rank == 0)cudaEventRecord(comp_s);
		FW_phase1<<< grid_phase1, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM);
           
		FW_phase2<<< grid_phase2, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM);
          
		FW_phase3<<< grid_phase3, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix, blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM, row_offset/blocking_factor);
         if(rank == 0){
			cudaEventRecord(comp_e);
			cudaEventSynchronize(comp_e);
			cudaEventElapsedTime(&t_e, comp_s, comp_e);
			comp_time += t_e;  
		}
	}
//	printf("[%d]allcoate finish. End FW\n",rank);	
	if(rank == 0) cudaEventRecord(mem_s);
	cudaMemcpy((void*) &(host_matrix[cpy_idx]), (void*) &(device_matrix[cpy_idx]), sizeof(int) * VERTEX_EXT*blocking_factor*num_blocks_per_thread, cudaMemcpyDeviceToHost);
	if(rank == 0){
		cudaEventRecord(mem_e);
		cudaEventSynchronize(mem_e);
		cudaEventElapsedTime(&t_m, mem_s, mem_e);
		memory_time += t_m;
	}
//	printf("\n");
	// N = vertex_num
	//end FW algorithm
	if(rank == 0) cudaEventRecord(comm_s);
	if (rank == 0) {
		int send_idx = 0;
		int send_cnt = VERTEX_EXT*blocking_factor*num_blocks_per_thread;
		int recv_idx = location(num_blocks_per_thread * blocking_factor, 0, VERTEX_EXT);
		int recv_cnt = VERTEX_EXT*blocking_factor*(num_blocks_per_thread + round % size);
/*		
		int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                int dest, int sendtag,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int source, int recvtag,
                MPI_Comm comm, MPI_Status *status)
*/				
		MPI_Sendrecv(&host_matrix[send_idx], send_cnt, MPI_INT, 1, 0, &host_matrix[recv_idx], recv_cnt, MPI_INT, 1, MPI_ANY_TAG,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else {
		int recv_idx = 0;
		int recv_cnt = VERTEX_EXT*blocking_factor*(num_blocks_per_thread - round % size);
		int send_idx = location((num_blocks_per_thread - round % size) * blocking_factor, 0, VERTEX_EXT);
		int send_cnt = VERTEX_EXT*blocking_factor*num_blocks_per_thread;
		MPI_Sendrecv(&host_matrix[send_idx], send_cnt, MPI_INT, 0, 0, &host_matrix[recv_idx], recv_cnt, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }
	  if(rank == 0){
			cudaEventRecord(comm_e);
			cudaEventSynchronize(comm_e);
			cudaEventElapsedTime(&t_c, comm_s, comm_e);
			comm_time += t_c; 
	  }
	MPI_Barrier(MPI_COMM_WORLD);
	// output
	
//	fh_out = fopen(OUTPUT_NAME,"w");
	if(rank == 0){
		//I/O
	cudaEventRecord(start);
		fh_out = fopen(OUTPUT_NAME,"w");
		for(int i = 0; i < vertex_num; i++) {
			for(int j = 0; j < vertex_num; j++) {
				if(host_matrix[i * VERTEX_EXT + j] >= INF) {
					fprintf(fh_out, "INF ");
				} else {
					fprintf(fh_out, "%d ", host_matrix[i * VERTEX_EXT + j]);
				}
			}
			fprintf(fh_out, "\n");
		}
		fclose(fh_out);
		//I/O
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	IO_time += t;
	}	
//	fclose(fh_out);


	//free memory
	cudaFreeHost(host_matrix);
	cudaFree(device_matrix);
/*	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);

	printf("Elapsed Time: %lf sec \n", elapsed_time/1000);
*/	
	printf("[exec] [I/O] [mem] [comp] [comm]\n");
	printf("%f %f %f %f %f\n", (IO_time+memory_time+comp_time+comm_time)/1000, IO_time/1000, memory_time/1000, comp_time/1000, comm_time/1000);
	MPI_Barrier(MPI_COMM_WORLD);		
	MPI_Finalize();
	return 0;
}