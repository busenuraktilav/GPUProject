
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


__global__ void subset_of_vertices(int *size, float *percentage, int *queue, int *subset_queue)
{
	int i = threadIdx.x + (blockDim.x * blockIdx.x);

	int t = (*size) * (*percentage);
	int temp = t;
	int rand = i*t + i + t + (*percentage) * (*size);

	if (i < t)
	{
		subset_queue[i] = (queue)[rand%(t)];
		//printf("%i\n", subset_queue[i]);
	}

	(*size) = temp;
}



extern "C"

void sbf(const int *row_ptr, const int *col_ind, const int *row_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source, float **appr_vals)
{

	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_nv;
	int *d_queue, *d_nextQueue, *d_nextSize, *d_iter, *d_subset_queue;
	float *d_percentage;
	int2* d_dist;


	int2* dist = (int2*)malloc(nv*sizeof(int2));


	int signal_partial_graph_process = (*appr_vals)[0];
	int signal_reduce_execution = (*appr_vals)[1];
	int iter_num = (*appr_vals)[2];
	float *percentage = (float*)malloc(nv*sizeof(float));
	*percentage = (*appr_vals)[3];



	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_dist, nv*sizeof(int2)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_subset_queue, (nv+1)*sizeof(int))); //space might be percentage*size 

	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));



	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	cudaProfilerStart();


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
    cudaCheck(cudaMalloc((void **)&d_percentage, sizeof(float)));
    
	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_dist, dist, nv*sizeof(int2), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_queue, srcArr, srcNeigh * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_percentage, percentage, sizeof(float), cudaMemcpyHostToDevice));


	int size = srcNeigh;
	int *nextSize = (int*)malloc(sizeof(int));
	*nextSize = 0;
	

	int round = 1;
	int temp = 0;
	

	// no approximation. Both signals are negative
	while((size > 0) && (round < nv) && temp < ne && !signal_reduce_execution && !signal_partial_graph_process) { temp += size;

		//printf("NO APPR\n");

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

		(*appr_vals)[2] = round;
		round++;

	}

	//printf("total size: %i\n", temp);

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
	printf("NEGATIVE CYCLE CHECK: %i\n", neg_cycle);

	*distance = (int*)malloc((nv+1)*sizeof(int)); 

	for (int i = 0; i < nv; i++)
	{
		(*distance)[i] = dist[i].x;
	}


	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_dist));


	printf("GPU SBF time(ms): %f\n", elapsed);
	
}