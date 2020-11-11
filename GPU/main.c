#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../graphio/graphio.h"

#define INF 9999

void sbf(const int *row_ptr, const int *col_ind, const int *row_ind, const int *weights, int *distance, int *previous, const int nv, const int ne, int source);
void sdj(const int *row_ptr, const int *col_ind, const int *weights, int *distance, int *previous, const int nv, const int ne, int source, int *visited);


int main(int argc, char const *argv[])
{
	const char* file = "../example_graphs/deneme.txt";
	//const char* file = "../example_graphs/EAT_RS.mtx";
	//const char* file = "../example_graphs/EAT_SR.mtx";
	//const char* file = "../example_graphs/wiki-RfA.mtx";

	int start = 0;

	int *row_ptr;
	int *col_ind;
	int *row_ind;
	int *weights;
	int nv;
	int ne;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne);

	int *distance = (int*)malloc(nv*sizeof(int)); 
	int *previous = (int*)malloc(nv*sizeof(int));
	int *visited = (int*)malloc(nv*sizeof(int));

	memset(distance, INF, sizeof(int) * nv);
	memset(previous, 0, sizeof(int) * nv);
	memset(visited, 0, sizeof(int) * nv);

	if(read == 1)
	{
		sbf(row_ptr, col_ind, row_ind, weights, distance, previous, nv, ne, start);
		//sdj(row_ptr, col_ind, weights, distance, previous, nv, ne, start, visited);
	}

	/*

	for (int i = 0; i<nv; i++){
		printf("distance[%i]: %i\n", i, distance[i]);
	}
	*/

	free(row_ptr);
	free(col_ind);
	free(row_ind);
	free(weights);
	free(distance);
	free(previous);
	free(visited);

	return 0;
}