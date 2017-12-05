#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

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
int** device_matrix;

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
	//time measurement
	
	float IO_time = 0.0;
	float memory_time = 0.0;
	float comp_time = 0.0;
	float t,t_m,t_e;
	cudaEvent_t start, stop;
	cudaEvent_t mem_s, mem_e;
	cudaEvent_t comp_s, comp_e;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&mem_s);
	cudaEventCreate(&mem_e);
	cudaEventCreate(&comp_s);
	cudaEventCreate(&comp_e);
	if (argc != 4) {
		fprintf(stderr, "Insuficcient arguments\n");
		fprintf(stderr, "Usage: ./%s input_file output_file blocking_factor\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	//load argument
	const char *INPUT_NAME = argv[1];
	const char *OUTPUT_NAME = argv[2];
	int blocking_factor = atoi(argv[3]);

	int num_devices = 1;
	cudaGetDeviceCount(&num_devices);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, omp_get_thread_num());
		printf("Dev%d. Name: %s\n", omp_get_thread_num(), prop.name);
	
	}


	// read file
	
	FILE *fh_in, *fh_out;
	fh_in = fopen(INPUT_NAME, "r");
	int edge_num, vertex_num;
	cudaEventRecord(start);
	fscanf(fh_in, "%d %d", &vertex_num, &edge_num);
	//I/O
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	IO_time += t;
	if (blocking_factor > vertex_num) blocking_factor = vertex_num;
	int VERTEX_EXT = vertex_num + (blocking_factor - ((vertex_num-1) % blocking_factor + 1));
 
	printf("Blocking factor: %d\n", blocking_factor);

	//allocate memory
	cudaMallocHost((void**) &host_matrix, sizeof(int) * VERTEX_EXT*VERTEX_EXT);
	device_matrix = (int**) malloc(sizeof(int*) * num_devices);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void**) &device_matrix[omp_get_thread_num()], sizeof(int) * VERTEX_EXT*VERTEX_EXT);
	}

	//initialize (store data in row major)
	for(int i = 0; i < VERTEX_EXT; ++i){
		for(int j = 0; j < VERTEX_EXT; ++j){
			if(i == j) host_matrix[i * VERTEX_EXT + j] = 0;
			else host_matrix[i * VERTEX_EXT + j] = INF;
		}
	}
	cudaEventRecord(start);
	int a, b, weight;//a and b is source_vertex and destination_vertex respectively 
	for(int i = 0; i < edge_num; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		a -= 1;
		b -= 1;
		host_matrix[a * VERTEX_EXT + b] = weight;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	IO_time += t;
	fclose(fh_in);
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

	#pragma omp parallel num_threads(num_devices)
	{
		int t_id = omp_get_thread_num();
		cudaSetDevice(t_id);

		int num_blocks_per_thread = round / num_devices;
		int row_offset = num_blocks_per_thread * t_id * blocking_factor;
		//allocate the remain num
		if (t_id == num_devices-1)
			num_blocks_per_thread += round % num_devices;

		dim3 grid_phase3(num_blocks_per_thread, round);
		
		
			
		
		int cpy_idx = location(row_offset, 0, VERTEX_EXT);
//		printf("[%d] num_thread: %d\n row_offset: %d\n cpy_idx: %d\n",t_id,num_blocks_per_thread,row_offset,cpy_idx);
		//copy from the beginning of the first pixel
		//mem
		if(t_id == 0) cudaEventRecord(mem_s);
		cudaMemcpy((void*) &(device_matrix[t_id][cpy_idx]), (void*) &(host_matrix[cpy_idx]), sizeof(int) * VERTEX_EXT*blocking_factor*num_blocks_per_thread, cudaMemcpyHostToDevice);
		//mem
		if(t_id == 0){
		cudaEventRecord(mem_e);
		cudaEventSynchronize(mem_e);
		cudaEventElapsedTime(&t_m, mem_s, mem_e);
		memory_time += t_m;
		}
		for (int r = 0; r < round; r++) {
            
			int r_idx = location(r * blocking_factor, 0, VERTEX_EXT);
			if(t_id == 0) cudaEventRecord(mem_s);
			if (r >= row_offset/blocking_factor && r < (row_offset/blocking_factor + num_blocks_per_thread)) {
//				printf("r=%d row_offset/blocking_factor=%d row_offset/blocking_factor + num_blocks_per_thread=%d\n",r,row_offset/blocking_factor,row_offset/blocking_factor + num_blocks_per_thread);
				cudaMemcpy((void*) &(host_matrix[r_idx]), (void*) &(device_matrix[t_id][r_idx]), sizeof(int) * VERTEX_EXT * blocking_factor, cudaMemcpyDeviceToHost);
			}
			if(t_id == 0){
			cudaEventRecord(mem_e);
			cudaEventSynchronize(mem_e);
			cudaEventElapsedTime(&t_m, mem_s, mem_e);
			memory_time += t_m;
			}
			#pragma omp barrier
			if(t_id == 0) cudaEventRecord(mem_s);
			cudaMemcpy((void*) &(device_matrix[t_id][r_idx]), (void*) &(host_matrix[r_idx]), sizeof(int) * VERTEX_EXT * blocking_factor, cudaMemcpyHostToDevice);
			if(t_id == 0){
			cudaEventRecord(mem_e);
			cudaEventSynchronize(mem_e);
			cudaEventElapsedTime(&t_m, mem_s, mem_e);
			memory_time += t_m;
			}
			if(t_id == 0) cudaEventRecord(comp_s);
			FW_phase1<<< grid_phase1, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix[t_id], blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM);
			cudaDeviceSynchronize();
			FW_phase2<<< grid_phase2, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix[t_id], blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM);
			cudaDeviceSynchronize();
			FW_phase3<<< grid_phase3, BLOCK_DIM, sizeof(int)*3*blocking_factor*blocking_factor >>>(device_matrix[t_id], blocking_factor, VERTEX_EXT, r, MAX_BLCOK_DIM, row_offset/blocking_factor);
			if(t_id == 0){
			cudaEventRecord(comp_e);
			cudaEventSynchronize(comp_e);
			cudaEventElapsedTime(&t_e, comp_s, comp_e);
			comp_time += t_e;
			}
           
		}
		if(t_id == 0) cudaEventRecord(mem_s);
		cudaMemcpy((void*) &(host_matrix[cpy_idx]), (void*) &(device_matrix[t_id][cpy_idx]), sizeof(int) * VERTEX_EXT*blocking_factor*num_blocks_per_thread, cudaMemcpyDeviceToHost);
		if(t_id == 0){
		cudaEventRecord(mem_e);
		cudaEventSynchronize(mem_e);
		cudaEventElapsedTime(&t_m, mem_s, mem_e);
		memory_time += t_m;
		}
		#pragma omp barrier
	}
	
//	printf("\n");
	// N = vertex_num
	//end FW algorithm
	

	
	// output
	//I/O
	cudaEventRecord(start);
	
	fh_out = fopen(OUTPUT_NAME,"w");
	for(int i = 0; i < vertex_num; ++i) {
		for(int j = 0; j < vertex_num; ++j) {
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
	//free memory
	cudaFreeHost(host_matrix);
	#pragma omp parallel num_threads(num_devices)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(device_matrix[omp_get_thread_num()]);
	}
	
	printf("[exec] [I/O] [mem] [comp]\n");
	printf("%f %f %f %f\n", (IO_time+memory_time+comp_time)/1000, IO_time/1000, memory_time/1000, comp_time/1000);
	
	return 0;
}