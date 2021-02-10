#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "hybrid.h"
#include "heap.h"
#include "../GPU/bellman_ford.cuh"
#include "../GPU/dijkstra.cuh"
#include "../GPU/hybrid.cuh"
//#include "cpu_utils.h" //not finished. will be added later


float relative_error(int **arr1, int **arr2, int size)
{
	float err = 0.0;
	float actual_val = 0.0;


	for (int i = 0; i < size; i++)
	{
		err += abs((*arr1)[i] - (*arr2)[i]);
		actual_val += (*arr1)[i];
	}

	return err/actual_val;

}

void init_zero(int **arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		(*arr)[i] = 0;
	}
}



int main(int argc, char const *argv[])
{
	const char* file = "../example_graphs/graph_19_16_pos-10.txt";
	
	int start;

	int *row_ptr, *col_ind, *row_ind, *weights, nv, ne, neg_edge_count = 0;
	int *gpu_bf_distance, *gpu_bf_previous;
	int *gpu_appr_dist1, *gpu_appr_dist2, *gpu_appr_prev1, *gpu_appr_prev2;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne, &neg_edge_count);


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

		
		clock_t strt, end;
		int cnt = 0;
	
		
		//BELLMAN-FORD GPU

		//appr_vals = [signal_partial_graph_process, signal_reduce_execution, iter_num, percentage]
		float *appr_vals = (float*)malloc(4*sizeof(float));

		

		sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start, &appr_vals);

		int iter_num = appr_vals[2];
		
		

		appr_vals[0] = 0;
		appr_vals[1] = 1; 

		do {

			iter_num--;

			appr_vals[2] = iter_num;

			sbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals);

			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

			init_zero(&gpu_appr_dist1, nv);

			printf("*******ERROR: %f\n", error);




		} while (iter_num > 0);



		appr_vals[0] = 1;
		appr_vals[1] = 0; 

		float percentage[9] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};

		for (int i = 0; i < 9; i++)
		{
			appr_vals[3] = percentage[i];
			printf("%f\n", percentage[i]);

			sbf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals);

			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

			init_zero(&gpu_appr_dist1, nv);

			printf("*******ERROR: %f\n", error);
		}




		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);
		
	}


	return 0;
}
