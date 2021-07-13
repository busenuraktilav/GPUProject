#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <omp.h>


#include "../../include/graphio/graphio.h"
#include "../../include/utils/cpu_utils.h"
#include "bellman_ford.h"

int main(int argc, char const *argv[])
{

	const char* distance_file = "../compare/bellman_originaldistance.txt";
	const char* time_results = "time_results.txt";
	const char* perf_results = "../../../analysis-results/bellman_performance_results.csv";

	const char* file = argv[1];
	
	int *row_ptr, *col_ind, *weights, *distance, max_weight, min_weight, nv, ne, 
	     iter_num, max_degree = 0, neg_edge_count = 0, start = 0;

	int read = read_graph(file, &row_ptr, &col_ind, &weights, &nv, &ne, &neg_edge_count, &max_weight, &min_weight);

	if(read != 1)
	{
		printf("Something went terribly wrong while reading the graph\n");
		return -1;
	}

	start = select_start_vertex(row_ptr, nv, &max_degree);

	printf("Start node: %i\n", start);
	printf("Max degree: %i\n", max_degree);

	double strt; 
	double end; 
	strt = omp_get_wtime(); 
	bellman_ford(&row_ptr, &col_ind, &weights, &distance, nv, ne, start);	
	end = omp_get_wtime(); 
	printf("Work took %f milliseconds\n", (end - strt)*1000);
	
	free(row_ptr);
	free(col_ind);
	free(weights);
	free(distance);
	

	return 0;

}
