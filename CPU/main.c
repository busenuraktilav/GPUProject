#include <stdio.h>
#include <stdlib.h>

#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "heap.h"


int main(int argc, char const *argv[])
{
	
	//const char* file = "./example_graphs/article.txt";
	//const char* file = "./example_graphs/deneme.txt";
	const char* file = "./example_graphs/football.mtx";

	int start = 0;

	int *row_ptr, *col_ind, *row_ind, *weights, nv, ne;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne);

	if(read == 1)
	{
		bellman_ford(&row_ptr, &col_ind, &row_ind, &weights, nv, ne, start);
		dijkstra(&row_ptr, &col_ind, &row_ind, &weights, nv, ne, start);
	}

	


	return 0;
}
