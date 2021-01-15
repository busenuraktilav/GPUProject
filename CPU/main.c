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
#include "../GPU/hybrid.cuh"
//#include "cpu_utils.h" //not finished. will be added later



int main(int argc, char const *argv[])
{
	const char* file = "../example_graphs/graph_13_16_pos-10.txt";
	//const char* file = "../example_graphs/article.txt";
	//const char* file = "../example_graphs/Trefethen_2000.mtx";
	//const char* file = "../example_graphs/graph_18_16_mix-1-100.txt";
	
	int start;

	int *row_ptr, *col_ind, *row_ind, *weights, nv, ne, neg_edge_count = 0;
	int *cpu_bf_distance, *cpu_bf_previous, *gpu_bf_distance, *gpu_bf_previous;
	int *cpu_dj_distance, *cpu_dj_previous, *gpu_dj_distance, *gpu_dj_previous;
	int *cpu_hybrid_distance, *cpu_hybrid_previous, *gpu_hybrid_distance, *gpu_hybrid_previous;

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
	
		
		//BELLMAN-FORD CPU codes run

		int num = 0;

		strt = clock();
		bellman_ford(&row_ptr, &col_ind, &row_ind, &weights, &cpu_bf_distance, &cpu_bf_previous, nv, ne, start);
		end = clock();
		double cpu_bf = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("BELLMAN-FORD CPU TIME (s): %f\n", cpu_bf);
	
		
		//BELLMAN-FORD GPU

		sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start);
	

		//DIJKSTRA GPU ALGORITHM

		int count;
		double gpu_dj = 0;

		sdj(row_ptr, col_ind, weights, &gpu_dj_distance, &gpu_dj_previous, nv, ne, start, &count, &gpu_dj);
	

		//DIJKSTRA CPU
			
		strt = clock();
		dijkstra(&row_ptr, &col_ind, &row_ind, &weights, &cpu_dj_distance, &cpu_dj_previous, nv, ne, start, &count);
		end = clock();
	
		double cpu_dj = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("DIJKSTRA CPU TIME (s): %f\n", cpu_dj);
		

		//HYBRID CPU CODE

		printf("neg_edge_count: %i\n", neg_edge_count);
		
		strt = clock();
		hybrid(&row_ptr, &col_ind, &row_ind, &weights, &cpu_hybrid_distance, &cpu_hybrid_previous, nv, ne, start, &count, neg_edge_count);
		end = clock();

		double cpu_hybrid = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("HYBRID CPU TIME (s): %f\n", cpu_hybrid);
		

		//HYBRID GPU CODE

		shybrid(row_ptr, col_ind, weights, &gpu_hybrid_distance, &gpu_hybrid_previous, nv, ne, start, &count, neg_edge_count);


		/*
		int check = 0;
		count = 0;

		printf("OK\n");


		for(int i = 0; i < nv; i++)
		{
			if(cpu_dj_distance[i] != gpu_dj_distance[i])
			//if(cpu_bf_distance[i] != gpu_bf_distance[i])

			//if(gpu_bf_distance[i] != gpu_hybrid_distance[i])
			//if(cpu_bf_distance[i] != gpu_hybrid_distance[i])
			{
				check = 1;
				count++;
				//printf("OK_IF\n");
				//printf("cpu_bf_distance[%i]: %i, gpu_bf_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_bf_distance[i]);
				printf("cpu_dj_distance[%i]: %i, gpu_dj_distance[%i]: %i\n", i, cpu_dj_distance[i], i, gpu_dj_distance[i]);
				//printf("gpu_bf_distance[%i]: %i, gpu_shybrid_distance[%i]: %i\n", i, gpu_bf_distance[i], i, gpu_hybrid_distance[i]);
				//printf("cpu_bf_distance[%i]: %i, gpu_shybrid_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_hybrid_distance[i]);
				//break;
			}
			else
			{
				//printf("OK_ELSE\n");
				//printf("cpu_bf_distance[%i]: %i, gpu_bf_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_bf_distance[i]);
				//printf("cpu_dj_distance[%i]: %i, gpu_dj_distance[%i]: %i\n", i, cpu_dj_distance[i], i, gpu_dj_distance[i]);
				//printf("cpu_bf_distance[%i]: %i, gpu_shybrid_distance[%i]: %i\n", i, cpu_bf_distance[i], i, gpu_hybrid_distance[i]);
			}
			//printf("inside for\n");
		}
		

		//printf("last ok\n");

		printf("check: %i, count: %i, nv: %i\n", check, count, nv);
		
		*/

		
		
		free(row_ptr);
		free(col_ind);
		free(row_ind);
		free(weights);
		


		free(cpu_bf_distance);
		free(cpu_dj_distance);
		//free(cpu_bf_previous);
		//free(cpu_dj_previous);
		
		free(gpu_bf_distance);
		free(gpu_dj_distance);
		//free(gpu_bf_previous);
		//free(gpu_dj_previous);
		
		
	}


	return 0;
}
