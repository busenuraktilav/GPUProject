#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>


#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "../GPU/bellman_ford.cuh"
#include "../GPU/dijkstra.cuh"
#include "cpu_utils.h"


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

min_edges_to_process
iter_num
percentage

*/


void main_bellman (int *row_ptr, int *col_ind, int *weights, int nv, int ne, 
	               int neg_edge_count, int max_weight, int min_weight, int max_degree,
	               int *signals, float *signal_variables, int start, bool write)
{

	const char* distance_file = "../bellman_originaldistance.txt";
	const char* time_results = "time_results.txt";
	const char* perf_results = "../bellman_performance_results.csv";
	
	int *distance, *approximate_distance;

	clock_t strt, end;
	int cnt = 0;
	float time = 0.0, error = 0.0;

	int iter_num;

	if (signals[0])
	{
		sbf(row_ptr, col_ind, weights, &distance, nv, ne, start, signals, 
			&signal_variables, &time);	
		write_distance(distance_file, distance, signal_variables[1], nv);
	}
		

	//Calculate the min edges 
	
	else if(signals[1] && !signals[3] && !signals[4])
	{
	    read_distance(distance_file, &distance, &iter_num, nv);
		signal_variables[0] = min_edge_to_process(row_ptr, nv, signal_variables[0]);
		
		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);

		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}
	
	//Calculate approximate values from technique4

	else if(signals[2])
	{
	    read_distance(distance_file, &distance, &iter_num, nv);

		int *level_arr = (int *)malloc((nv) * sizeof(int));
		int level = sync_bfs(row_ptr, col_ind, nv, ne, level_arr, start);
		approximate_attributes(&weights, ne, &max_weight, &min_weight);
		int max_distance = level * max_weight;
		
		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, max_distance, &time);
		
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
    }


	//Calculate the reduced execution from technique1

	else if (signals[3] && !signals[4] && !signals[1])
	{
		int temp_iter;
	    read_distance(distance_file, &distance, &temp_iter, nv);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);

		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}
	
	// Calculate the processing part of the graph from technique2

	else if(signals[4] && !signals[1] && !signals[3])
	{
	    read_distance(distance_file, &distance, &iter_num, nv);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
	
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}

	else if (signals[5] || signals[6] || signals[7])
	{
	    read_distance(distance_file, &distance, &iter_num, nv);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
		
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("*******ERROR: %f\n", error);
	}

	else if (signals[1] && signals[3] && !signals[4])
	{
		int temp_iter;
	    read_distance(distance_file, &distance, &temp_iter, nv);

		signal_variables[0] = min_edge_to_process(row_ptr, nv, signal_variables[0]);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
	
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}

	else if (signals[1] && signals[4] && !signals[3])
	{
	    read_distance(distance_file, &distance, &iter_num, nv);

		signal_variables[0] = min_edge_to_process(row_ptr, nv, signal_variables[0]);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
	
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}

	else if (signals[4] && signals[3] && !signals[1])
	{
		int temp_iter;
	    read_distance(distance_file, &distance, &temp_iter, nv);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
		
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);
	}


	else if (signals[1] && signals[3] && signals[4])
	{
		int temp_iter;
	    read_distance(distance_file, &distance, &temp_iter, nv);
		signal_variables[0] = min_edge_to_process(row_ptr, nv, signal_variables[0]);

		apprbf(row_ptr, col_ind, weights, &approximate_distance, nv, ne, start, signals,
		       &signal_variables, INT_MAX, &time);
	
		error = relative_error(&distance, &approximate_distance, nv);
		init_zero(&approximate_distance, nv);
		printf("******* ERROR *******: %f\n", error);

	}


	if (write)
	{
		write_performance_results(perf_results, time_results, nv, ne, max_degree, 
			                      signals, signal_variables, error);
	}
	
	else
	{
		write_time_results(time_results, time);
	}


	free(row_ptr);
	free(col_ind);
	free(weights);
}
