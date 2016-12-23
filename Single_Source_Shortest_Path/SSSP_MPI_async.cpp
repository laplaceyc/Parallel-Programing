#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stack>
#define MAX 2147364847
#define INF -1
#define false 0
#define true 1
#define BELLMAN_TAG 0
#define TOKEN_TAG 1
#define TERMINATION_TAG 2
#define black 0
#define white 1

int main (int argc, char *argv[]) {
	int rank, size;
	MPI_Status status;
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
	//close file
	fclose(fh_in);
	
	//start bellmanford algorithm
	int token = white;
	//initial condition
	int distance = (rank == source) ? 0 : MAX;
	int parent = (rank == source) ? source : -1;
	int engage = 0;
	//engage bellmanford algorithm
	if(rank == source){
		for(int i = 0; i < vertex_num; i++){
			if(i == rank){ continue; }
			else if(local_adj_matrix[i] != INF){
				if(i < rank) token = black;
				int new_distance = distance + local_adj_matrix[i];
			//	int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
				MPI_Send(&new_distance, 1, MPI_INT, i, BELLMAN_TAG, MPI_COMM_WORLD);
				printf("Send done! new distance = %d destination = %d\n", new_distance, i);
			}
		}
	}
		
	int package;//special delivery!!

	//	int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
    //               MPI_Comm comm, MPI_Status *status)
	while(MPI_Recv(&package, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status) == MPI_SUCCESS){
		//printf("[rank %d] Recvive done! pakage tag = %d package content = %d from = %d\n", rank, status.MPI_TAG, package, status.MPI_SOURCE);
		//open package and check
		if(status.MPI_TAG == BELLMAN_TAG){
			//package is a distance
			int recv_distance = package;
			//compare with distance and recv_distance
			if(recv_distance < distance){
				//if recv_distance is smaller than renew data
				parent = status.MPI_SOURCE;
				distance = recv_distance;
				//acknowledge other neightbor
				for(int i = 0; i < vertex_num; i++){
					if(i == rank){ continue; }
					//if there is a path
					else if(local_adj_matrix[i] != INF){
						if(i < rank) token = black;
						int new_distance = distance + local_adj_matrix[i];
						//	int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
						MPI_Send(&new_distance, 1, MPI_INT, i, BELLMAN_TAG, MPI_COMM_WORLD);
					}
				}
			}
			
			//engage dual-pass ring termination algorithm 
			if(rank == 0 && engage == 0){
				MPI_Send(&token, 1, MPI_INT, 1, TOKEN_TAG, MPI_COMM_WORLD);
				engage = 1;
				
			}
		}else if(status.MPI_TAG == TOKEN_TAG){
			//package is a token
			token = package;
			if(rank == 0){
				if(token == white){ 
					MPI_Send(&token, 1, MPI_INT, 1, TERMINATION_TAG, MPI_COMM_WORLD);
					break;
				}else{//token is black, a new round begin
					token = white;
					MPI_Send(&token, 1, MPI_INT, 1, TOKEN_TAG, MPI_COMM_WORLD);
				}
			}else{//otherwise pass to next processor
				token = token & package;
				MPI_Send(&token, 1, MPI_INT, (rank + 1) % size, TOKEN_TAG, MPI_COMM_WORLD);
			}
		}else if(status.MPI_TAG == TERMINATION_TAG){
			//package is a termination signal
			MPI_Send(&token, 1, MPI_INT, (rank + 1) % size, TERMINATION_TAG, MPI_COMM_WORLD);
			break;
		}
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
	fclose(fh_out);
	delete [] root_buf;
	MPI_Finalize();

	return 0;
		
}