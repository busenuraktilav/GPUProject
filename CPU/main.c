#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../graphio/graphio.h"
#include "bellman_ford.h"
#include "dijkstra.h"
#include "heap.h"
#include "../GPU/bellman_ford.cuh"
#include "../GPU/dijkstra.cuh"
//#include "cpu_utils.h" //not finished. will be added later



int main(int argc, char const *argv[])
{
	
	
	//const char* file = "../example_graphs/article.txt";
	//const char* file = "../example_graphs/deneme.txt";
	const char* file = "../example_graphs/football.mtx";
	//const char* file = "../example_graphs/EAT_RS.mtx";
	//const char* file = "../example_graphs/EAT_SR.mtx";
	//const char* file = "../example_graphs/internet.mtx";
	//const char* file = "../example_graphs/Linux_call_graph.mtx";
	//const char* file = "../example_graphs/wiki-RfA.mtx";
	//const char* file = "../example_graphs/bcsstm01.mtx";
	//const char* file = "../example_graphs/psmigr_1.mtx";
	//const char* file = "../example_graphs/Trefethen_150.mtx";
	//const char* file = "../example_graphs/Trefethen_200b.mtx";
	//const char* file = "../example_graphs/Trefethen_300.mtx";
	//const char* file = "../example_graphs/Trefethen_500.mtx";
	//const char* file = "../example_graphs/Trefethen_700.mtx";
	//const char* file = "../example_graphs/Trefethen_2000.mtx";
	//const char* file = "../example_graphs/Trefethen_20000.mtx";


	int start = 0;

	int *row_ptr, *col_ind, *row_ind, *weights, *cpu_bf_distance, *gpu_bf_distance, *cpu_dj_distance, *gpu_dj_distance, 
	    *cpu_bf_previous, *gpu_bf_previous, *cpu_dj_previous, *gpu_dj_previous, *visited, nv, ne;

	int read = read_graph(file, &row_ptr, &col_ind, &row_ind, &weights, &nv, &ne);

	if(read == 1)
	{
		clock_t strt, end;

		//CPU codes run
		strt = clock();
		bellman_ford(&row_ptr, &col_ind, &row_ind, &weights, &cpu_bf_distance, &cpu_bf_previous, nv, ne, start);
		end = clock();
		double cpu_bf = ((double) (end - strt)) / CLOCKS_PER_SEC;


		strt = clock();
		dijkstra(&row_ptr, &col_ind, &row_ind, &weights, &cpu_dj_distance, &cpu_dj_previous, nv, ne, start);
		end = clock();
		double cpu_dj = ((double) (end - strt)) / CLOCKS_PER_SEC;

		printf("CPU_BF: %f\nCPU_DJ: %f\n", cpu_bf, cpu_dj);


		//GPU codes run
		sbf(row_ptr, col_ind, row_ind, weights, &gpu_bf_distance, &gpu_bf_previous, nv, ne, start);
		sdj(row_ptr, col_ind, weights, &gpu_dj_distance, &gpu_dj_previous, nv, ne, start);


		//distance check. write a func later

		int check = 1;
		int count = 0;

		for(int i = 0; i < nv; i++)
		{
			if(cpu_dj_distance[i] != gpu_dj_distance[i])
			//if(cpu_bf_distance[i] != gpu_bf_distance[i])
			{
				check = 0;
				count++;
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
		free(cpu_bf_distance);
		free(cpu_dj_distance);
		free(cpu_bf_previous);
		free(cpu_dj_previous);

	}

	return 0;
}
