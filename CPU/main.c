#include <stdio.h>
#include <stdlib.h>


#include "bellman_ford.h"
#include "dijkstra.h"
#include "hybrid.h"
#include "cpu_utils.h"
#include "../graphio/graphio.h"

/*

### signal order in signal array ###

signal_originalDistance [0]
signal_kernelMinEdge [1]
signal_appr_attr [2]
signal_reduce_execution [3]
signal_partial_graph_process [4]
signal_atomicMinBlock [5]
signal_atomicMaxBlock [6]
signal_atomicAddBlock [7]


### the variable order in signal_variables array ###

min_edges_to_process [0]
iter_num [1]
percentage [2]

*/

int main(int argc, char const *argv[])
{
	// Take the input variables with argument to figure out which algorithm will run 
	int *signals = (int *) malloc ( 8 * sizeof(int));
	float *signal_variables = (float *) malloc (3 * sizeof(float));
	

	for (int i = 0; i < 8; ++i)
	{
		signals[i] = atoi(argv[i+1]);
	}


	for (int i = 0; i < 3; ++i)
	{
		signal_variables[i] = atof(argv[i + 11]);
	}

	const char* file = argv[10];
	
	bool write = atoi(argv[14]);
	int algorithm_option = atoi(argv[15]);


	int *row_ptr, *col_ind, *weights, max_weight, min_weight, nv, ne, deg, max_degree = 0, neg_edge_count = 0, start = 0;
	int read = read_graph(file, &row_ptr, &col_ind, &weights, &nv, &ne, &neg_edge_count, &max_weight, &min_weight);

	if(read != 1)
	{
		printf("Something went terribly wrong while reading the graph\n");
		return -1;
	}

	// Select the starting vertex (vertex that contains the max num of edge)
	for (int i = 0; i < nv; ++i)
	{
		deg = row_ptr[i + 1] - row_ptr[i];
		if (deg > max_degree)
		{
			start = i;
			max_degree = deg;
		}
	}

	printf("Start node: %i\n", start);
	printf("Max degree: %i\n", max_degree);

	switch (algorithm_option) 
	{
		case 1:

			main_bellman(row_ptr, col_ind, weights, nv, ne, neg_edge_count, max_weight,
			             min_weight, max_degree, signals, signal_variables, start, write);

			break;

		case 2:

			main_dijkstra(row_ptr, col_ind, weights, nv, ne, neg_edge_count, max_weight,
			              min_weight, max_degree, signals, signal_variables, start, write);
			break;

		case 3:

			main_hybrid(row_ptr, col_ind, weights, nv, ne, neg_edge_count, max_weight,
			            min_weight, max_degree, signals, signal_variables, start, write);
			break;

		default:

			printf("Algorithm_option is not passed to main function!\n");
			break;
	}

	return 0;
}
