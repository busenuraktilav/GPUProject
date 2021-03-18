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


int main(int argc, char const *argv[])
{
	//const char* file = "../example_graphs/graph_20_16_pos-10.txt";
	//const char* file = "../example_graphs/Trefethen_150.mtx";
	const char* file = argv[6];
	const char* distance_file = "../dijkstra_originaldistance.txt";
	const char* perf_results = "../dijkstra_performance_results.txt";

	
	
	int start;

	int *row_ptr, *col_ind, *row_ind, *weights, max_weight, min_weight, nv, ne, neg_edge_count = 0;
	int *gpu_dj_distance, *gpu_dj_previous;
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


		bool signal_originalDistance = atoi(argv[1]);
		bool signal_kernelMinEdge = atoi(argv[2]);
		bool signal_appr_attr = atoi(argv[3]);
		bool signal_reduce_execution = atoi(argv[4]);
		int signal_partial_graph_process = atoi(argv[5]);

		printf("%i, %i, %i, %i, %i\n", signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, signal_reduce_execution, signal_partial_graph_process);



		//appr_vals = [signal_partial_graph_process, signal_reduce_execution, iter_num, percentage, min_edges_to_process]
		float *appr_vals = (float*)malloc(5*sizeof(float));
		
		if (signal_originalDistance)
		{

			appr_vals[0] = 0;
			appr_vals[1] = 0;
			int min_edge = 0;
			float percentage = 100.0;

			apprsdj(row_ptr, col_ind, weights, &gpu_dj_distance, &gpu_dj_previous, nv, ne, start, &appr_vals, INT_MAX, &time);
	
			int iter_num = appr_vals[2];
			printf("iter_num: %i\n", iter_num);
			
			write_distance(distance_file, gpu_dj_distance, &iter_num, &max_degree, nv);


			write_performance_results(perf_results, nv, ne, iter_num, max_degree, 
	                          min_edge, percentage, signal_originalDistance, 
	                          signal_kernelMinEdge, signal_appr_attr, signal_reduce_execution, 
	                          signal_partial_graph_process, 0, time);


		    
		}
			

		//Calculate the min edges 
		
		if(signal_kernelMinEdge)
		{
			int iter_num;
		    read_distance(distance_file, &gpu_dj_distance, &iter_num, &max_degree, nv);

			appr_vals[0] = 0;
			appr_vals[1] = 0;
			int p[9];
			float percentage = 100.0;
			
			 
			if (max_degree < 10000){

				for (int i = 1; i < 10; ++i)
					p[i-1] = (max_degree/100)*i;
			}

			else if (max_degree < 100000 && max_degree >= 10000){
				for (int i = 1; i < 10; ++i)
					p[i-1] = (max_degree/1000)*i*2;
			}

			else {

				for (int i = 1; i < 10; ++i)
					p[i-1] = (max_degree/10000)*i*4;
			}

			for (int i = 0; i < 9; i++)
			{
				appr_vals[4] = p[i];
				printf("%d\n", p[i]);

				apprsdj(row_ptr, col_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, INT_MAX, &time);

				float error = relative_error(&gpu_dj_distance, &gpu_appr_dist3, nv);

				init_zero(&gpu_appr_dist3, nv);

				printf("*******ERROR: %f\n", error);

				write_performance_results(perf_results, nv, ne, iter_num, max_degree, 
	                          p[i], percentage, signal_originalDistance, 
	                          signal_kernelMinEdge, signal_appr_attr, signal_reduce_execution, 
	                          signal_partial_graph_process, error, time);

			}
			
			/*
			appr_vals[4] = 10;

			apprsdj(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, INT_MAX, &time);

			float error = relative_error(&gpu_dj_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);
			*/
		}
		
		//Calculate approximate values from technique4

		if(signal_appr_attr)
		{
			int min_edge = 0;
			float percentage = 100.0;

			int iter_num;
		    read_distance(distance_file, &gpu_dj_distance, &iter_num, &max_degree, nv);

			int *level_arr = (int *)malloc((nv) * sizeof(int));
			int level = sync_bfs(row_ptr, col_ind, nv, ne, level_arr, start);

			approximate_attributes(&weights, ne, &max_weight, &min_weight);

			int max_distance = level * max_weight;
			
			appr_vals[0] = 0;
			appr_vals[1] = 0; 

			apprsdj(row_ptr, col_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, max_distance, &time);
			
			
			float error = relative_error(&gpu_dj_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);

			write_performance_results(perf_results, nv, ne, iter_num, max_degree, 
	                          min_edge, percentage, signal_originalDistance, 
	                          signal_kernelMinEdge, signal_appr_attr, signal_reduce_execution, 
	                          signal_partial_graph_process, error, time);
	    
	    }


		//Calculate the reduced execution from technique1

		if (signal_reduce_execution)
		{
			int min_edge = 0;
			float percentage = 100.0;

			int iter_num;
		    read_distance(distance_file, &gpu_dj_distance, &iter_num, &max_degree, nv);
			
			appr_vals[0] = 0;
			appr_vals[1] = 1;

			int i = 1;

			printf("iter_num: %i\n", iter_num);

			do {

				appr_vals[2] = iter_num;

				apprsdj(row_ptr, col_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals, INT_MAX, &time);

				float error = relative_error(&gpu_dj_distance, &gpu_appr_dist1, nv);

				init_zero(&gpu_appr_dist1, nv);

				printf("*******ERROR: %f\n", error);

				write_performance_results(perf_results, nv, ne, iter_num, max_degree, 
	                          min_edge, percentage, signal_originalDistance, 
	                          signal_kernelMinEdge, signal_appr_attr, signal_reduce_execution, 
	                          signal_partial_graph_process, error, time);

				iter_num--;

			} while (iter_num >= 0);
		}
		
		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);

		
	}


	return 0;
}
