#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stack>
#include <sys/time.h>
#define MAX 2147364847
#define INF -1

struct Node{
	int v;//vertex
	int id;//thread_id
};

pthread_mutex_t mutex;
static int thread_num;
int vertex_num, edge_num, min, c;
int **adjacency_matrix, *distance, *parent;
bool *visit;//true means visiting done, otherwise false

void *relaxation(void *data){
	Node *my_data = (Node *) data;
	for(int i = my_data -> id; i < vertex_num; i += thread_num){
		//if there is no path from v to i
		if(!visit[i] && adjacency_matrix[my_data -> v][i] == INF) { continue; }
		//else if judge the new path cost, renew path while costing less 
		else if(!visit[i] && distance[my_data -> v] + adjacency_matrix[my_data -> v][i] < distance[i]){
			distance[i] = distance[my_data -> v] + adjacency_matrix[my_data -> v][i];
			parent[i] = my_data -> v;
		}
	}
	pthread_exit(NULL);
}

int main (int argc, char *argv[]) {
	//check type of argument
	if (argc != 5) {
		fprintf(stderr, "Insuficcient arguments\n");
		fprintf(stderr, "Usage: ./%s thread input_file output_file source_id\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	//load argument
	thread_num = atoi(argv[1]);
	const char *INPUT_NAME = argv[2];
	const char *OUTPUT_NAME = argv[3];
	const int source = atoi(argv[4]) - 1;
	
	//load vertex&edge information
	FILE *fh_in, *fh_out;
	fh_in = fopen(INPUT_NAME,"r");
	if(fh_in == NULL){
		printf("Input file open failed.\n");
	}
	fscanf(fh_in,"%d %d",&vertex_num,&edge_num);
	if(vertex_num - 1 > edge_num){
		fprintf(stderr, "Input graph is not a connected graph\n");
		fprintf(stderr, "Error on type of input graph\n");
		exit(EXIT_FAILURE);
	}
	
	//dynamic allocate memory to adjacency_matrix
	adjacency_matrix = new int*[vertex_num];
	for(int i = 0; i < vertex_num; i++) adjacency_matrix[i] = new int[vertex_num];
	
	//initialize matrix
	for(int i = 0; i < vertex_num; i++)
		for(int j = 0; j < vertex_num; j++)
			adjacency_matrix[i][j] = INF;
			
	//load weight
	int a, b, weight;//a and b is vertex_id1 and vertex_id2 respectively 
	for(int i = 0; i < edge_num; i++){
		fscanf(fh_in, "%d %d %d", &a, &b, &weight);
		adjacency_matrix[a - 1][b - 1] = weight;
		adjacency_matrix[b - 1][a - 1] = weight;
	}
	fclose(fh_in);
	//dynamic allocate memory to distance, parent and visit
	distance = new int[vertex_num];
	parent = new int[vertex_num];
	visit = new bool[vertex_num];
	
	//initialize array
	for(int i = 0; i < vertex_num; i++){
		distance[i] = MAX;
		parent[i] = -1;
		visit[i] = false;
	}
	
	//initial condition
	distance[source] = 0;
	parent[source] = source;
	
	for(int i = 0; i < vertex_num; i++){
		min = MAX, c = -1;
		pthread_t threads[thread_num];
		Node data[thread_num];
		
		for(int i = 0; i < vertex_num; i++){
			if(!visit[i] && distance[i] < min){
			c = i;
			min = distance[i];
			}
		}
		
		if(c == -1) break;
		visit[c] = true;
		
		for(int j = 0; j < thread_num; j++){
			data[j].id = j;
			data[j].v = c;
			pthread_create(&threads[j], NULL, relaxation, (void *)&data[j]);
		}
		
		for(int j = 0; j < thread_num; j++){
			pthread_join(threads[j], NULL);
		}
	}
	
	//return pointer
	for(int i = 0; i < vertex_num; i++) delete [] adjacency_matrix[i];
	delete [] adjacency_matrix;

	//write file
	fh_out = fopen(OUTPUT_NAME,"w");
	if(fh_out == NULL){
		printf("Output file open failed.\n");
	}
	std::stack<int> S;
	for(int i = 0; i < vertex_num; i++){
		int in_stack = i;
		S.push(in_stack);
		while(S.top() != source){			
			in_stack = parent[in_stack];
			S.push(in_stack);
		}
		fprintf(fh_out,"%d", source + 1);
	
		while(!S.empty()){
			if(S.top() == source && i != source) {
				S.pop();
				continue;
			}
			fprintf(fh_out," %d", S.top() + 1);
			S.pop();
		}
		fprintf(fh_out,"\n");
	}
	fclose(fh_out);
	//return pointer
	delete [] distance;
	delete [] parent;
	delete [] visit;
			
	pthread_exit(NULL);
}