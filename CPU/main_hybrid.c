#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "hybrid.h"
#include "heap.h"
#include "../GPU/bellman_ford.cuh"
#include "../GPU/dijkstra.cuh"
#include "../GPU/hybrid.cuh"
#include "cpu_utils.h"


int main_hybrid(bool signal_originalDistance, bool signal_kernelMinEdge, bool signal_appr_attr, 
	            bool signal_reduce_execution, int signal_partial_graph_process, const char *file,
	            float min_edge, float iter_num, float percentage, bool write)
{
	
	const char* distance_file = "../hybrid_originaldistance.txt";
	const char* time_results = "time_results.txt";
	const char* perf_results = "../hybrid_performance_results.csv";
	
	int start;

	int *row_ptr, *col_ind, *row_ind, *weights, max_weight, min_weight, nv, ne, neg_edge_count = 0;
	int *gpu_hybrid_distance, *gpu_hybrid_previous;
	int *gpu_appr_dist1, *gpu_appr_dist2, *gpu_appr_dist3, *gpu_appr_prev1, *gpu_appr_prev2, *gpu_appr_prev3;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne, &neg_edge_count, &max_weight, &min_weight);




	if(read == 1)
	{

		int start = 0;
		int max_degree = 0, deg;
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

		
		clock_t strt, end;
		int cnt = 0;
		float time = 0;

		//appr_vals = [signal_partial_graph_process, signal_reduce_execution, iter_num, percentage, min_edges_to_process]
		float *appr_vals = (float*)malloc(8*sizeof(float));
		/*
		appr_vals[4] = min_edges_to_process;
		appr_vals[2] = iter_num;
		appr_vals[3] = percentage;
		*/

		// TEMPORARILY

		bool signal_atomicMinBlock = 0;
		bool signal_atomicMaxBlock = 0;
		bool signal_atomicAddBlock = 0;

		appr_vals[0] = signal_partial_graph_process;
		appr_vals[1] = signal_reduce_execution;
		appr_vals[2] = iter_num;
		appr_vals[3] = percentage;
		appr_vals[2] = iter_num;
		appr_vals[4] = 0;
		appr_vals[5] = signal_atomicMinBlock;
		appr_vals[6] = signal_atomicMaxBlock;
		appr_vals[7] = signal_atomicAddBlock;

		
		if (signal_originalDistance)
		{

			appr_vals[0] = 0;
			appr_vals[1] = 0;
			float percentage = 1.0;
			float error = 0.0;

			apprshybrid(row_ptr, col_ind, weights, &gpu_hybrid_distance, &gpu_hybrid_previous, nv, ne, start, neg_edge_count, &appr_vals, INT_MAX, &time);
	
			int iter_num = appr_vals[2];
			printf("iter_num: %i\n", iter_num);
			
			write_distance(distance_file, gpu_hybrid_distance, &iter_num, &max_degree, nv);


			if (write)
			{
				write_performance_results(perf_results, time_results, nv, ne, iter_num, max_degree, 
							   min_edge, percentage, signal_originalDistance, signal_kernelMinEdge, 
							   signal_appr_attr, signal_reduce_execution, signal_partial_graph_process,
							   signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock, error);
			}
			
			else
			{
				write_time_results(time_results, time);
			}
		    
		}
			

		//Calculate the min edges 
		
		if(signal_kernelMinEdge)
		{
			int iter_num;
		    read_distance(distance_file, &gpu_hybrid_distance, &iter_num, &max_degree, nv);

			appr_vals[0] = 0;
			appr_vals[1] = 0;
			float percentage = 1.0;

			appr_vals[4] = min_edge_to_process(row_ptr, nv, min_edge);

			apprshybrid(row_ptr, col_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, neg_edge_count, &appr_vals, INT_MAX, &time);

			float error = relative_error(&gpu_hybrid_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);

			if (write)
			{
				write_performance_results(perf_results, time_results, nv, ne, iter_num, max_degree, 
							   appr_vals[4], percentage, signal_originalDistance, signal_kernelMinEdge, 
							   signal_appr_attr, signal_reduce_execution, signal_partial_graph_process,
							   signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock, error);
			
			}
			
			else
			{
				write_time_results(time_results, time);
			}
			
		}
		
		//Calculate approximate values from technique4

		if(signal_appr_attr)
		{
			float percentage = 1.0;

			int iter_num;
		    read_distance(distance_file, &gpu_hybrid_distance, &iter_num, &max_degree, nv);

			int *level_arr = (int *)malloc((nv) * sizeof(int));
			int level = sync_bfs(row_ptr, col_ind, nv, ne, level_arr, start);

			approximate_attributes(&weights, ne, &max_weight, &min_weight);

			int max_distance = level * max_weight;
			
			appr_vals[0] = 0;
			appr_vals[1] = 0; 

			apprshybrid(row_ptr, col_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, neg_edge_count, &appr_vals, max_distance, &time);
			
			
			float error = relative_error(&gpu_hybrid_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);

			if (write)
			{
				write_performance_results(perf_results, time_results, nv, ne, iter_num, max_degree, 
							   min_edge, percentage, signal_originalDistance, signal_kernelMinEdge, 
							   signal_appr_attr, signal_reduce_execution, signal_partial_graph_process,
							   signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock, error);
			}
			
			else
			{
				write_time_results(time_results, time);
			}

	    }


		//Calculate the reduced execution from technique1

		if (signal_reduce_execution)
		{
			float percentage = 1.0;

			int iter;
		    read_distance(distance_file, &gpu_hybrid_distance, &iter, &max_degree, nv);
			
			appr_vals[0] = 0;
			appr_vals[1] = 1;


			apprshybrid(row_ptr, col_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, neg_edge_count, &appr_vals, INT_MAX, &time);

			float error = relative_error(&gpu_hybrid_distance, &gpu_appr_dist1, nv);

			init_zero(&gpu_appr_dist1, nv);

			printf("*******ERROR: %f\n", error);

			if (write)
			{
				write_performance_results(perf_results, time_results, nv, ne, iter_num, max_degree, 
							   min_edge, percentage, signal_originalDistance, signal_kernelMinEdge, 
							   signal_appr_attr, signal_reduce_execution, signal_partial_graph_process,
							   signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock, error);
			}
			
			else
			{
				write_time_results(time_results, time);
			}

		}
		
		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);

		
	}


	return 0;
}
