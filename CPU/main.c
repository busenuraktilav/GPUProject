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
//#include "cpu_utils.h" //not finished. will be added later



float relative_error(int **arr1, int **arr2, int size)
{
	float err = 0.0;
	float actual_val = 0.0;
	float error = 0.0;


	for (int i = 0; i < size; i++)
	{
		if(((*arr1)[i] == INT_MAX && (*arr2)[i]) || ((*arr2)[i] == INT_MAX && (*arr1)[i]))
		{
			//error += 1;
		}
		else 
		{
			err = abs((*arr1)[i] - (*arr2)[i]);
			actual_val = (*arr1)[i];
			if(!isnan(err/actual_val))
				error += err/actual_val;
		}
	}

	return (error/size)*100;

}

/*

float relative_error(int **arr1, int **arr2, int size)
{
	float err = 0.0;
	float actual_val = 0.0;


	for (int i = 0; i < size; i++)
	{
		err += abs((*arr1)[i] - (*arr2)[i]);
		actual_val += (*arr1)[i];

	}

	return (err/actual_val)*100;

}
*/


void init_zero(int **arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		(*arr)[i] = 0;
	}
}


//Technique 4: round weights to the nearest power of 2
void approximate_attributes(int **weights, int ne, int *max_weight, int *min_weight)
{
	*max_weight = pow(2,round(log2(*max_weight)));
	*min_weight = pow(2,round(log2(*min_weight)));

	int k = log2(*max_weight) - log2(*min_weight);

	//int max_distance = level * (*max_weight);

	for (int i = 0; i < ne; i++)
	{
		(*weights)[i] = pow(2,round(log2((*weights)[i])));
	}
}


int sync_bfs(int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source)
{
    memset(level_arr, -1, nv * sizeof(int));
    int level = 0;
    level_arr[source] = level;
    bool improvement = true;

    while (improvement)
    {
        improvement = false;
        for (int v = 0; v < nv; ++v)
        {
            if (level_arr[v] == level)
            {
                for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; ++edge)
                {
                    int adj = col_ind[edge];
                    if (level_arr[adj] == -1)
                    {
                        improvement = true;
                        level_arr[adj] = level + 1;
                    }
                }
            }
        }
        level++;
    }
    return level;
}



int main(int argc, char const *argv[])
{
	const char* file = "../example_graphs/graph_21_16_pos-10.txt";
	
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

		
		clock_t strt, end;
		int cnt = 0;


		//BELLMAN-FORD GPU

		//appr_vals = [signal_partial_graph_process, signal_reduce_execution, iter_num, percentage]
		float *appr_vals = (float*)malloc(4*sizeof(float));

		
		//appr_vals[0] = 0;
		//appr_vals[1] = 1;

		//sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start, &appr_vals);



		//Calculate the actual distance
		appr_vals[0] = 0;
		appr_vals[1] = 0; 

		denemebf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start, &appr_vals, INT_MAX);
		
		

		
		//Calculate approximate values from technique4
		int *level_arr = (int *)malloc((nv) * sizeof(int));
		int level = sync_bfs(row_ptr, col_ind, nv, ne, level_arr, start);

		approximate_attributes(&weights, ne, &max_weight, &min_weight);

		int max_distance = level * max_weight;


		appr_vals[0] = 0;
		appr_vals[1] = 0; 

		denemebf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist3, &gpu_appr_prev3, nv, ne, start, &appr_vals, max_distance);
		
		
		float error = relative_error(&gpu_bf_distance, &gpu_appr_dist3, nv);

		init_zero(&gpu_appr_dist3, nv);

		printf("*******ERROR: %f\n", error);


		/*

		//Calculate the reduced execution from technique1

		int iter_num = appr_vals[2];
		//int iter_num = 3;
		
		//sleep(15);

		appr_vals[0] = 0;
		appr_vals[1] = 1; 

		do {

			iter_num--;

			appr_vals[2] = iter_num;

			denemebf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals, INT_MAX);

			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

			init_zero(&gpu_appr_dist1, nv);

			printf("*******ERROR: %f\n", error);

			//sleep(10);


		} while (iter_num > 0);
		*/

		
		/*

		// Calculate the processing part of the graph from technique2

		appr_vals[0] = 1;
		appr_vals[1] = 0; 

		float percentage[9] = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1};

		for (int i = 0; i < 9; i++)
		{
			appr_vals[3] = percentage[i];
			printf("%f\n", percentage[i]);

			denemebf(row_ptr, col_ind, row_ind, weights, &gpu_appr_dist1, &gpu_appr_prev1, nv, ne, start, &appr_vals, INT_MAX);

			float error = relative_error(&gpu_bf_distance, &gpu_appr_dist1, nv);

			init_zero(&gpu_appr_dist1, nv);

			printf("*******ERROR: %f\n", error);
		}

		*/


		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);
		
	}


	return 0;
}
