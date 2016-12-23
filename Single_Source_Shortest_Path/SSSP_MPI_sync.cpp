#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stack>
#define MAX 2147364847
#define INF -1
#define false 0
#define true 1
#define BELLMAN_TAG 0

int main (int argc, char *argv[]) {
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//check type of argument
	if (argc != 5) {
		fprintf(stderr, "Insuficcient arguments\n");
		fprintf(stderr, "Usage: ./%s thread input_file output_file source_id\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	//load argument
//	thread_num = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];
	const int source = atoi(argv[4]) - 1;
	
	FILE *fh_in, *fh_out;
	int vertex_num, edge_num;
	fh_in = fopen(INPUT_NAME,"r");
	if(fh_in == NULL && rank == source){
		printf("Input file open failed.\n");
	}
	fscanf(fh_in,"%d %d",&vertex_num,&edge_num);
	if(vertex_num != size){
		fprintf(stderr, "Error on specification of vertex centric\n");
		fprintf(stderr, "MPI processes must equal to vertex num\n");
		exit(EXIT_FAILURE);
	}
	if(vertex_num - 1 > edge_num){
		fprintf(stderr, "Error on type of input graph\n");
		fprintf(stderr, "Input graph is not a connected graph\n");
		exit(EXIT_FAILURE);
	}

	//partition each row of adjacent matrix to each processor 
	int *local_adj_matrix = new int[vertex_num];
	for(int i = 0; i < vertex_num; i++) local_adj_matrix[i] = INF;
	//load weight
	int a, b, weight;//a and b is vertex_id1 and vertex_id2 respectively 
	for(int i = 0; i < edge_num; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		a -= 1;
		b -= 1;
		if(a == rank) local_adj_matrix[b] = weight;
		if(b == rank) local_adj_matrix[a] = weight;
	}
//	fclose(fh_in);
	//start bellmanford algorithm
	int sorted = false, all_sorted = false;
	//initial condition
	int distance = (rank == source) ? 0 : MAX;
	int parent = (rank == source) ? source : -1;
	
	while (!all_sorted) {
		sorted = true;
		//send distance information
		for(int i = 0; i < vertex_num; i++){
			if(i == rank){ continue; }
			else if(local_adj_matrix[i] != INF){
				int new_distance = distance + local_adj_matrix[i];
				int recv_distance;
			//	int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            //  int dest, int sendtag,
            //  void *recvbuf, int recvcount, MPI_Datatype recvtype,
            //  int source, int recvtag,
            //  MPI_Comm comm, MPI_Status *status)
				MPI_Sendrecv(&new_distance, 1, MPI_INT, i, BELLMAN_TAG, &recv_distance, 1, MPI_INT, i, BELLMAN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if(recv_distance < distance){
					distance = recv_distance;
					parent = i;
					sorted = false;
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&sorted, &all_sorted, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
	}
	delete [] local_adj_matrix;
//	int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
// 				   void *recvbuf, int recvcount, MPI_Datatype recvtype,
//                 int root, MPI_Comm comm)
	int *root_buf = new int[vertex_num];
	MPI_Gather(&parent, 1, MPI_INT, root_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

	fh_out = fopen(OUTPUT_NAME,"w");
	if(fh_out == NULL){
		printf("Output file open failed.\n");
	}
	if(rank == 0){
		std::stack<int> S;
//		printf("Output File:\n");
		for(int i = 0; i < vertex_num; i++){
			int in_stack = i;
			S.push(in_stack);
			while(S.top() != source){			
				in_stack = root_buf[in_stack];
				S.push(in_stack);
			}
			
//			printf("%d", source + 1);
			fprintf(fh_out,"%d", source + 1);
			
			while(!S.empty()){
				if(S.top() == source && i != source) {
					S.pop();
					continue;
				}
//				printf(" %d", S.top() + 1);
				fprintf(fh_out," %d", S.top() + 1);
				S.pop();
			}
//			printf("\n");
			fprintf(fh_out,"\n");
		}
	}
//	fclose(fh_out);
	delete [] root_buf;
	MPI_Finalize();

	return 0;
		
}