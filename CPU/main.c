#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "hybrid.h"
#include "heap.h"
#include "../GPU/bellman_ford.cuh"
#include "../GPU/dijkstra.cuh"
//#include "cpu_utils.h" //not finished. will be added later



int main(int argc, char const *argv[])
{
	//const char* file = "../example_graphs/graph_10_16_pos-10.txt";
	//const char* file = "../example_graphs/deneme.txt";
	const char* file = "../example_graphs/Trefethen_2000.mtx";
	
	int start = 1;

	int *row_ptr, *col_ind, *row_ind, *weights, *cpu_bf_distance, *gpu_bf_distance, *cpu_dj_distance, *gpu_dj_distance, 
	    *cpu_hybrid_distance, *gpu_hybrid_distance, *cpu_bf_previous, *gpu_bf_previous, *cpu_dj_previous, *gpu_dj_previous, 
	    *cpu_hybrid_previous, *gpu_hybrid_previous, *visited, nv, ne;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne);


	if(read == 1)
	{
		clock_t strt, end;

		printf("before bellman_ford\n");

		//CPU codes run
		strt = clock();
		bellman_ford(&row_ptr, &col_ind, &row_ind, &weights, &cpu_bf_distance, &cpu_bf_previous, nv, ne, start);
		end = clock();
		double cpu_bf = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("after bellman_ford\n");

		strt = clock();
		dijkstra(&row_ptr, &col_ind, &row_ind, &weights, &cpu_dj_distance, &cpu_dj_previous, nv, ne, start);
		end = clock();
		double cpu_dj = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("after dijkstra\n");

		strt = clock();
		hybrid(&row_ptr, &col_ind, &row_ind, &weights, &cpu_hybrid_distance, &cpu_hybrid_previous, nv, ne, start);
		end = clock();
		double cpu_hybrid = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("after hybrid\n");

		printf("CPU_BF: %f\nCPU_DJ: %f\nCPU_HYBRID: %f\n", cpu_bf, cpu_dj, cpu_hybrid);


		//GPU codes run
		sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start);
		sdj(row_ptr, col_ind, weights, &gpu_dj_distance, &gpu_dj_previous, nv, ne, start);


		//distance check. write a func later

		int check = 0;
		int count = 0;

		for(int i = 0; i < nv; i++)
		{
			//if(cpu_dj_distance[i] != gpu_dj_distance[i])
			if(cpu_bf_distance[i] != gpu_bf_distance[i])
			{
				check = 1;
				//count++;
				//printf("cpu_bf_distance[%i]: %i, gpu_bf_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_bf_distance[i]);
				//printf("cpu_dj_distance[%i]: %i, gpu_dj_distance[%i]: %i\n", i, cpu_dj_distance[i], i, gpu_dj_distance[i]);
				//break;
			}
			else
			{
				//printf("cpu_bf_distance[%i]: %i, gpu_bf_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_bf_distance[i]);
				//printf("cpu_dj_distance[%i]: %i, gpu_dj_distance[%i]: %i\n", i, cpu_dj_distance[i], i, gpu_dj_distance[i]);
			}
		}

		printf("check: %i, count: %i, nv: %i\n", check, count, nv);

		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);
		//free(cpu_bf_distance);
		free(cpu_dj_distance);
		//free(cpu_bf_previous);
		free(cpu_dj_previous);
		free(gpu_bf_distance);
		free(gpu_dj_distance);
		free(gpu_bf_previous);
		free(gpu_dj_previous);
		
	}

	return 0;
}
