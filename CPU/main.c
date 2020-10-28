#include <stdio.h>
#include <stdlib.h>

#include "graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "heap.h"


int main(int argc, char const *argv[])
{
	
	//const char* file = "article.txt";
	const char* file = "deneme.txt";
	//const char* file = "football.mtx";

	int start = 1;

	int *row_ptr, *col_ind, *row_ind, *weights, nv, ne;

	read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne);

	bellman_ford(&row_ptr, &col_ind, &row_ind, &weights, nv, ne, start);


	return 0;
}
