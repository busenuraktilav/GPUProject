#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#include "../../../include/graphio/graphio.h"
#include "../../../include/utils/cpu_utils.h"
#include "dijkstra.cuh"


int main (int argc, char const *argv[])
{

	const char* distance_file = "../../compare/dijkstra_originaldistance.txt";
	const char* time_results = "time_results.txt";
	const char* error_results = "error_results.txt";
	const char* perf_results = "../../../../analysis-results/dijkstra_performance_results.csv";

	const char* file = argv[1];
	int write = atoi(argv[2]);

	int *row_ptr, *col_ind, *weights, *distance, *approximate_distance, max_weight, min_weight, nv, ne, 
	     iter_num = 0, min_edges = 0, max_degree = 0, neg_edge_count = 0, start = 0;

	float time = 0.0, error = 0.0, percentage = 1.0;

	int read = read_graph(file, &row_ptr, &col_ind, &weights, &nv, &ne, &neg_edge_count, &max_weight, &min_weight);

	if(read != 1)
	{
		printf("Something went terribly wrong while reading the graph\n");
		return -1;
	}

	read_distance(distance_file, &distance, &iter_num, nv);
	start = select_start_vertex(row_ptr, nv, &max_degree);

	printf("Start node: %i\n", start);
	printf("Max degree: %i\n", max_degree);

	time = apprsdj(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start);
	error = relative_error(&distance, &approximate_distance, nv);

	printf("******* ERROR *******: %f\n", error);

	int *signals = (int *) calloc (9 , sizeof(int));
	signals[5] = 1;
	
	if (write)
	{
		write_time_results(time_results, time);
		write_time_results(error_results, error);
		write_performance_results(perf_results, time_results, nv, ne, max_degree, 
			                      signals, min_edges, max_degree, iter_num, percentage, error);
		if (remove(time_results) != 0)
		{
			printf("ATTENTION: time_results file could not be deleted!");
		}
		if (remove(error_results) != 0)
		{
			printf("ATTENTION: time_results file could not be deleted!");
		}
	}
	
	else
	{
		write_time_results(time_results, time);
		write_time_results(error_results, error);
	}


	free(row_ptr);
	free(col_ind);
	free(weights);
	free(distance);

	return 0;
}
