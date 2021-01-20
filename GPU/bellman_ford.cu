
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

#define N_THREADS_PER_BLOCK 128


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
    	du = distance[u];

        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
            
            v = col_ind[e];
            w = weights[e];
            dv = distance[v];
            add = du + w;

            if (add < dv && du != INT_MAX)
            {
            	atomicMin(&distance[v],add);
            	index = atomicAdd(nextSize, 1);
				nextQueue[index] = v;
            }

        }
    }

}

__global__ void relaxAtom(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int2* distance, int *iter) 
{

    int index, u, v, w, du, dv, add, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < size) { 
        
    	u = queue[tid];
    	du = distance[u].x;

        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
            
            v = col_ind[e];
            w = weights[e];
            dv = distance[v].x;
            add = du + w;

            if (add < dv && du != INT_MAX)
            {
            	atomicMin(&distance[v].x,add);

            	if(distance[v].y != *iter)
            	{
	            	atomicMax(&distance[v].y,*iter);
	            	index = atomicAdd(nextSize, 1);
					nextQueue[index] = v;
				}
            }
        }
    }
}

/*
__global__ void relaxInit(int *row_ptr, int *col_ind, int *weights, int *source, int *nextQueue, int* size, int* nextSize, int2* distance)

{
    int index, v, w, du, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 
        
    	du = distance[*source].x;
    	//printf("du %i\n", du);

        for (int e = row_ptr[*source]; e < row_ptr[(*source) + 1]; e++) {
            
            v = col_ind[e];
            w = weights[e];
            atomicMin(&distance[v].x,du+w);

            //if (distance[v].y != 1)
            //{
	        	
	        	index = atomicAdd(nextSize, 1);
				nextQueue[index] = v;

				//printf("distance %i\n", v);
			//}

        }
    }
    __syncthreads();
}*/



__global__ void initVar(int2 *distance, int *nv)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;

	for (int k = i; k < *nv; k += skip) {

		distance[k].x = INT_MAX;
		distance[k].y = 1; // dont update this in relaxinit. update here. no need for mem copy
	}

	__syncthreads();

}



extern "C"

void sbf(const int *row_ptr, const int *col_ind, const int *row_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source)
{

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	cudaProfilerStart();

	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_nv;
	int *d_queue, *d_nextQueue, *d_nextSize, *d_iter;
	int2* d_dist;


	int2* dist = (int2*)malloc(nv*sizeof(int2));


	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_dist, nv*sizeof(int2)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));

	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));


	initVar<<<(nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_dist, d_nv);

	cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));

	//To increase parallelism, first process the source vertex
	int srcNeigh = row_ptr[source + 1] - row_ptr[source];
	int *srcArr = (int*)calloc(srcNeigh, sizeof(int));
	int index = 0;
	dist[source].x = 0;

	for (int i = row_ptr[source]; i < row_ptr[source + 1]; i++) { // for each neighbor of source node
		
		int w = weights[i];
		int du = dist[source].x;
		int dv = dist[col_ind[i]].x;

		if (du + w < dv && du != INT_MAX)
		{
			dist[col_ind[i]].x = du + w;
			dist[col_ind[i]].y = 1;
			srcArr[index++] = col_ind[i]; // add to frontier
		}
	}

	int *iter = (int*)malloc(sizeof(int));
	*iter = 2;


	// Allocate device
	
	cudaCheck(cudaMalloc((void **)&d_nextSize, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_queue, (nv+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void **)&d_nextQueue, (nv+1)*sizeof(int)));
    cudaCheck(cudaMalloc((void **)&d_iter, sizeof(int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_dist, dist, nv*sizeof(int2), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_queue, srcArr, srcNeigh * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
	


	int size = srcNeigh;
	int *nextSize = (int*)malloc(sizeof(int));
	*nextSize = 0;
	

	int round = 1;
	int temp = 0;


	while((size > 0) && (round < nv) && temp < ne) { temp += size;

		cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));

		cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

		relaxAtom<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter);

		cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

		(*iter) ++;


		printf("size: %i\n", size);

		size = *nextSize;
		*nextSize = 0;
		std::swap(d_queue, d_nextQueue); // swap frontiers

		//printf("round: %i\n", round);

		round++;

	}

	printf("total size: %i\n", temp);

	// Copy outputs to host
	cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	
	cudaProfilerStop();

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	float elapsed;
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));


	// check for negative cycles
	int neg_cycle = false;

	for (int v2 = 0; v2 < nv; v2++)
	{
		for (int e = (row_ptr)[v2]; e < (row_ptr)[v2+1]; e++)
		{
			int tempDistance = dist[v2].x + (weights)[e];
			int neigh = (col_ind)[e];

			if (tempDistance < dist[neigh].x && dist[v2].x != INT_MAX)
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
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_dist));



	printf("GPU SBF time(ms): %f\n", elapsed);
	
}