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
	const char* file = "../example_graphs/graph_15_16_pos-10.txt";
	//const char* file = "../example_graphs/Trefethen_150.mtx";
	const char* distance_file = "../originaldistance.txt";

	
	
	int start;

	int *row_ptr, *col_ind, *row_ind, *weights, max_weight, min_weight, nv, ne, neg_edge_count = 0;
	int *gpu_bf_distance, *gpu_bf_previous;
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


		bool signal_originalDistance = 1;
		bool signal_kernelMinEdge = 0;
		bool signal_appr_attr = 0;
		bool signal_reduce_execution = 0;
		bool signal_partial_graph_process = 0;



		//BELLMAN-FORD GPU

		//appr_vals = [signal_partial_graph_process, signal_reduce_execution, iter_num, percentage, min_edges_to_process]
		float *appr_vals = (float*)malloc(5*sizeof(float));
		
		if (signal_originalDistance)
		{

			appr_vals[0] = 0;
			appr_vals[1] = 0;

			sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start, &appr_vals);
			
			int iter_num = appr_vals[2];
			printf("iter_num: %i\n", iter_num);
			
			write_distance(distance_file, gpu_bf_distance, &iter_num, nv);
		    
		}
			

		//Calculate the min edges 
		
		if(signal_kernelMinEdge)
		{

			appr_vals[0] = 0;
			appr_vals[1] = 0; 
			appr_vals[4] = 100; 
			
			apprbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, INT_MAX);
			
			
			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);
		}
		
		//Calculate approximate values from technique4

		if(signal_appr_attr)
		{
			int *level_arr = (int *)malloc((nv) * sizeof(int));
			int level = sync_bfs(row_ptr, col_ind, nv, ne, level_arr, start);

			approximate_attributes(&weights, ne, &max_weight, &min_weight);

			int max_distance = level * max_weight;
			
			appr_vals[0] = 0;
			appr_vals[1] = 0; 

			apprbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, max_distance);
			
			
			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist3, nv);

			init_zero(&gpu_appr_dist3, nv);

			printf("*******ERROR: %f\n", error);
	    
	    }


		//Calculate the reduced execution from technique1

		if (signal_reduce_execution)
		{
			int iter_num;
		    read_distance(distance_file, &gpu_bf_distance, &iter_num, nv);
			
			appr_vals[0] = 0;
			appr_vals[1] = 1;

			int i = 1;

			printf("iter_num: %i\n", iter_num);

			do {

				iter_num--;

				appr_vals[2] = iter_num+1;

				apprbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals, INT_MAX);

				float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

				init_zero(&gpu_appr_dist1, nv);

				printf("*******ERROR: %f\n", error);

			} while (iter_num >= 0);
		}
		

		
		// Calculate the processing part of the graph from technique2

		if(signal_partial_graph_process)
		{
			appr_vals[0] = 1;
			appr_vals[1] = 0; 

			float percentage[9] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};

			for (int i = 0; i < 9; i++)
			{
				appr_vals[3] = percentage[i];
				printf("%f\n", percentage[i]);

				apprbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals, INT_MAX);

				float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

				init_zero(&gpu_appr_dist1, nv);

				printf("*******ERROR: %f\n", error);
			}
		}
		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);
		
	}


	return 0;
}
