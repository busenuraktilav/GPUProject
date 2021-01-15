
extern "C" {
#include "bellman_ford.cuh"
}

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#define N_THREADS_PER_BLOCK 1024


#define cudaCheck(error) \
	if (error != cudaSuccess) { \
    	printf("Fatal error: %s at %s:%d\n", \
    		cudaGetErrorString(error), \
    		__FILE__, __LINE__); \
    	exit(1); \
 }

__global__ void relax(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int* nv, int* distance) 
{

    int index, u, v, w, du, dv, add, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < size) { 
        
    	u = queue[tid];
    	//printf("queue[%i]: %i\n", tid, queue[tid]);
    	du = distance[u];

    	//printf("row_ptr[%i]: %i, row_ptr[%i]: %i\n", u, row_ptr[u], u+1, row_ptr[u+1]);

        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
            
            v = col_ind[e];
            w = weights[e];
            dv = distance[v];
            add = du + w;
            //printf("tid:%i, u: %i, du: %i, v: %i, w: %i, dv: %i, add:%i\n",tid, u, du, v, w, dv,add );

            if (du == INT_MAX){
                add = INT_MAX;
            }

            if (add < dv)
            {
            	atomicMin(&distance[v],add);
            	index = atomicAdd(nextSize, 1);
				nextQueue[index] = v;
            }

        }
    }

    if(tid == size)
    {

    }
}


int
compare_ints (const void *a, const void *b)
{
  const int *da = (const int *) a;
  const int *db = (const int *) b;

  return (int)(*da > *db) - (*da < *db);
}


extern "C"

void sbf(const int *row_ptr, const int *col_ind, const int *row_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source)
{

	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_row_ind, *d_weights, *d_distance, *d_previous, *d_nv, *d_ne;
	int *d_queue, *d_nextQueue, *d_nextSize;
	
	// Initialize CPU variables
	*distance = (int*)malloc(nv*sizeof(int)); 
	*previous = (int*)malloc(nv*sizeof(int));
	int *visited = (int *)calloc(nv, sizeof(int));


	for (int i = 0; i < nv; i++)
	{
		(*distance)[i] = INT_MAX;
		(*previous)[i] = -1;
	}

	(*distance)[source] = 0;
	visited[source] = 1;


	//To increase parallelism, first process the source vertex
	int srcNeigh = row_ptr[source + 1] - row_ptr[source];
	int *srcArr = (int*)calloc(srcNeigh, sizeof(int));
	int index = 0;
	(*distance)[source] = 0;

	for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) { // for each neighbor of source node
		
		int w = weights[i];
		int du = (*distance)[source];
		int dv = (*distance)[col_ind[i]];

		if (du + w < dv && du != INT_MAX)
		{
			(*distance)[col_ind[i]] = du + w;
			srcArr[index++] = col_ind[i]; // add to frontier
			//printf("col_ind[i]: %i\n", col_ind[i]);
		}
	}


	// Allocate device
	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_row_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_distance, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_previous, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void**) &d_nextSize, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_ne, sizeof(int)));
	cudaCheck(cudaMalloc((void **) &d_queue, (nv+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void **) &d_nextQueue, (nv+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void **) &d_nextSize, sizeof(int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_row_ind, row_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_distance, (*distance), nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_queue, srcArr, srcNeigh * sizeof(int), cudaMemcpyHostToDevice));


	int size = srcNeigh;
	int *nextSize = (int*)malloc(sizeof(int));
	*nextSize = 0;
	
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	int round = 1;

	cudaProfilerStart();

	int temp = 0;

	while((size > 0) && (round < nv)) { temp += size;

		cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

		relax<<<(size + 1023) / 1024, 1024>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_nv, d_distance);

		cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

		printf("size: %i\n", size);

		size = *nextSize;
		*nextSize = 0;
		std::swap(d_queue, d_nextQueue); // swap frontiers

		//printf("round: %i\n", round);

		round++;

	}

	printf("total size: %i\n", temp);

	cudaProfilerStop();

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	float elapsed;
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

	// Copy outputs to host
	cudaCheck(cudaMemcpy((*distance), d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy((*previous), d_previous, nv*sizeof(int), cudaMemcpyDeviceToHost));


	// check for negative cycles
	int neg_cycle = false;

	for (int v2 = 0; v2 < nv; v2++)
	{
		for (int e = (row_ptr)[v2]; e < (row_ptr)[v2+1]; e++)
		{
			int tempDistance = (*distance)[v2] + (weights)[e];
			int neigh = (col_ind)[e];

			if (tempDistance < (*distance)[neigh] && (*distance)[v2] != INT_MAX)
			{
				printf("GPU belman ford Error: negative cycle exists\n");
				neg_cycle = true;
				break;
			}

		}
		if(neg_cycle)
			break;
	}


	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_row_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_distance));
	cudaCheck(cudaFree(d_previous));

	printf("GPU SBF time: %f\n", elapsed/1000);

	/*
	for(int i = 0; i < nv; i++)
	{
		if((*distance)[i] != INT_MAX)
			printf("%i\n", (*distance)[i]);

	}
	*/
	
}