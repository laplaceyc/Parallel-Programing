#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define INF 10000000


int *host_matrix;
int *device_matrix;

__device__ int judge(int a, int b, int c){
	return (a > b + c) ? b + c : a;
} 

__global__ void FW_phase1(int b, int round, int n, int* din, int elements) {

	int x_end = round * b + b;
	int y_end = round * b + b;
	#pragma unroll
	for(int k = 0; k < elements; k++) {
		for(int j = 0; j < elements; j++) {
			int x = round * b + threadIdx.x + j * 32;
			int y = round * b + threadIdx.y + k * 32;
			if(x < n && y < n && x < x_end && y < y_end) {
				for(int i = round * b; i < (round + 1) * b && i < n; i++) {
					din[y * n + x] = judge(din[y * n + x], din[y * n + i], din[i * n + x]);
					__syncthreads();
				}

			}
		}

	}

}

__global__ void FW_phase2(int b, int round, int n, int* din, int elements) {

	int x_end, y_end;
	if (blockIdx.x == round) {
		return;
	}

	if(blockIdx.y == 0) {
		x_end = blockIdx.x * b + b;
		y_end = round * b + b;
		#pragma unroll
		for(int k = 0; k < elements; k++) {
			for(int j = 0; j < elements; j++) {
				int x = blockIdx.x * b + threadIdx.x + j * 32;
				int y = round * b + threadIdx.y + k * 32;
				if(x < n && y < n && x < x_end && y < y_end) {
					for(int i = round * b; i < (round + 1) * b && i < n; i++) {
						din[y * n + x] = judge(din[y * n + x], din[y * n + i], din[i * n + x]);
						__syncthreads();
					}

				}
			}

		}

	} else {
		// column
		x_end = round * b + b;
		y_end = blockIdx.x * b + b;
		#pragma unroll
		for(int k = 0; k < elements; k++) {
			for(int j = 0; j < elements; j++) {
				int x = round * b + threadIdx.x + j * 32;
				int y = blockIdx.x * b + threadIdx.y + k * 32;
				if(x < n && y < n && x < x_end && y < y_end) {
					for(int i = round * b; i < (round + 1) * b && i < n; i++) {
						din[y * n + x] = judge(din[y * n + x], din[y * n + i], din[i * n + x]);
					__syncthreads();
					}

				}
			}

		}

	}

}

__global__ void FW_phase3(int b, int round, int n, int *din, int elements) {

	int block_col = blockIdx.x >= round ? blockIdx.x + 1 : blockIdx.x;
	int block_row = blockIdx.y >= round ? blockIdx.y + 1 : blockIdx.y;

	int x_end = block_col * b + b;
	int y_end = block_row * b + b;

	#pragma unroll
	for(int k = 0; k < elements; k++) {
		for(int j = 0; j < elements; j++) {
			int x = block_col * b + threadIdx.x +  j * 32;
			int y = block_row * b + threadIdx.y +  k * 32;
			if(x < n && y < n && x < x_end && y < y_end) {
				for(int i = round * b; i < (round + 1) * b && i < n; i++) {
					din[y * n + x] = judge(din[y * n + x], din[y * n + i], din[i * n + x]);
			__syncthreads();
				}

			}
		}

	}

}


int main(int argc, char *argv[]) {
    
	if (argc != 4) {
		fprintf(stderr, "Insuficcient arguments\n");
		fprintf(stderr, "Usage: ./%s input_file output_file blocking_factor\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	//load argument
	const char *INPUT_NAME = argv[1];
	const char *OUTPUT_NAME = argv[2];
	int blocking_factor = atoi(argv[3]);
	
	// read file
	FILE *fh_in, *fh_out;
	int vertex_num, edge_num;
	
	fh_in = fopen(INPUT_NAME,"r");
	fscanf(fh_in,"%d %d",&vertex_num,&edge_num);
	host_matrix = new int[vertex_num * vertex_num]();
	//initialize (store data in row major)
	for(int i = 0; i < vertex_num; i++)
		for(int j = 0; j < vertex_num; j++)
			if(i != j) host_matrix[i * vertex_num + j] = INF;

	int a, b, weight;//a and b is source_vertex and destination_vertex respectively 
	for(int i = 0; i < edge_num; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		a -= 1;
		b -= 1;
		host_matrix[a * vertex_num + b] = weight;
	}	
	fclose(fh_in);
	
	
	// cuda
	int num_devices;
	int dev_index = 0;
	cudaDeviceProp prop;



	cudaGetDeviceCount(&num_devices);
	cudaGetDeviceProperties(&prop, dev_index);
	printf("Dev. Name: %s\n", prop.name);
	printf("Dev. Max threads per block: %d\n", prop.maxThreadsPerBlock);
	//time measurement
/*	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
*/	
	// init device
	cudaSetDevice(0);
	
	//start FW algorithm
	if(blocking_factor > vertex_num) blocking_factor = vertex_num;
	//Get the ceiling of round
	int round = (vertex_num + blocking_factor - 1) / blocking_factor;
	//Determine block dimension
	int MAX_BLCOK_DIM = blocking_factor > 32 ? 32 : blocking_factor;
	//Get the ceiling of element per thread
	int ELEMENT_PER_THREAD = blocking_factor > 32 ? (blocking_factor + 32 - 1) / 32 : 1;
	//Declare 3D parameter
	dim3 BLOCK_DIM(MAX_BLCOK_DIM, MAX_BLCOK_DIM);
	
	int blocks[3];
    //phase 1
	blocks[0] = blocking_factor;
	dim3 grid_phase1(1);
	// phase 2
	blocks[1] = (vertex_num + blocking_factor - 1) / blocking_factor;
	dim3 grid_phase2(blocks[1], 2);
	// phase 3
	blocks[2] = (vertex_num + blocking_factor - 1) / blocking_factor - 1;
	dim3 grid_phase3(blocks[2], blocks[2]);


    // init device
	cudaSetDevice(0);

//	int *device_matrix = NULL;
	cudaMalloc((void**) &device_matrix, vertex_num * vertex_num * sizeof(int));

	// compute
	cudaMemcpy(device_matrix, host_matrix, vertex_num * vertex_num * sizeof(int), cudaMemcpyHostToDevice);
	for(int r = 0; r < round; r++) {
		
		// phase 1
		FW_phase1 <<< grid_phase1, BLOCK_DIM >>> (blocking_factor, r, vertex_num, device_matrix, ELEMENT_PER_THREAD);
		cudaDeviceSynchronize();

		// phase2
		FW_phase2 <<< grid_phase2, BLOCK_DIM >>> (blocking_factor, r, vertex_num, device_matrix, ELEMENT_PER_THREAD);
		cudaDeviceSynchronize();

		// phase3
		FW_phase3 <<< grid_phase3, BLOCK_DIM >>> (blocking_factor, r, vertex_num, device_matrix, ELEMENT_PER_THREAD);
	}
	cudaMemcpy(host_matrix, device_matrix, vertex_num * vertex_num * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_matrix);
	
	
	//end FW algorithm
    // output
	fh_out = fopen(OUTPUT_NAME,"w");
	for(int i = 0; i < vertex_num; i++) {
		for(int j = 0; j < vertex_num; j++) {
			if(host_matrix[i * vertex_num + j] >= INF) {
				fprintf(fh_out, "INF ");
			} else {
				fprintf(fh_out, "%d ", host_matrix[i * vertex_num + j]);
			}
		}
		fprintf(fh_out, "\n");
	}
	fclose(fh_out);
	cudaFreeHost(host_matrix);
	return 0;
}
