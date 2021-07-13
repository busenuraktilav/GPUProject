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

extern "C"

__global__ void relax(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                  int *nextQueue, int size, int* nextSize, int* nv, int* distance) 
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

__global__ void relaxAtomicOperations(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                  int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                  int *iter) 
{

    int index, u, v, w, du, dv, add, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 
        
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
					//printf("kernel next size: %i\n", nextSize);
				}
            }
        }
    }
}


__global__ void relaxAtomicOperationsBlock(int *row_ptr, int *col_ind, int *weights, 
                                           int *queue, int *nextQueue, int *size, int* nextSize, 
                                           int2* distance, int *iter) 
{

    int index, u, v, w, du, dv, add, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 
        
    	u = queue[tid];
    	du = distance[u].x;

        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
            
            v = col_ind[e];
            w = weights[e];
            dv = distance[v].x;
            add = du + w;

            if (add < dv && du != INT_MAX)
            {
            	atomicMin_block(&distance[v].x,add);

            	if(distance[v].y != *iter)
            	{
	            	atomicMax_block(&distance[v].y,*iter);
	            	index = atomicAdd_block(nextSize, 1);
					nextQueue[index] = v;
				}
            }
        }
    }
}



__global__ void cudarelaxAtomicLessEdges(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *max_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]);

    	if ((edge_num < *max_edges)) // process only the nodes which has more than ... edges 
    	{
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
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}


__global__ void cudarelaxAtomicLessEdgesBlock(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *max_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]);

    	if ((edge_num < *max_edges)) // process only the nodes which has more than ... edges 
    	{
	        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
	            
	            v = col_ind[e];
	            w = weights[e];
	            dv = distance[v].x;
	            add = du + w;

	            if (add < dv && du != INT_MAX)
	            {
	            	atomicMin_block(&distance[v].x,add);

	            	if(distance[v].y != *iter)
	            	{
		            	atomicMax_block(&distance[v].y,*iter);
		            	index = atomicAdd_block(nextSize, 1);
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}


__global__ void cudarelaxAtomicMoreEdges(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *min_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]) ;

    	if (edge_num > *min_edges) // process only the nodes which has more than ... edges 
    	{
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
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}


__global__ void cudarelaxAtomicMoreEdgesBlock(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *min_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]) ;

    	if (edge_num > *min_edges) // process only the nodes which has more than ... edges 
    	{
	        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
	            
	            v = col_ind[e];
	            w = weights[e];
	            dv = distance[v].x;
	            add = du + w;

	            if (add < dv && du != INT_MAX)
	            {
	            	atomicMin_block(&distance[v].x,add);

	            	if(distance[v].y != *iter)
	            	{
		            	atomicMax_block(&distance[v].y,*iter);
		            	index = atomicAdd_block(nextSize, 1);
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}


__global__ void cudarelaxAtomicMoreLessEdges(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *min_edges, int *max_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]);

    	if ((edge_num > *min_edges) && (edge_num < *max_edges)) // process only the nodes which has more than ... edges 
    	{
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
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}


__global__ void cudarelaxAtomicMoreLessEdgesBlock(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int *size, int* nextSize, int2* distance, 
	                                     int *iter, int *min_edges, int *max_edges) 
{

    int index, u, v, w, du, dv, add, edge_num, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < *size) { 

    	u = queue[tid];
    	du = distance[u].x;
    	edge_num = (row_ptr[u+1] - row_ptr[u]);

    	if ((edge_num > *min_edges) && (edge_num < *max_edges)) // process only the nodes which has more than ... edges 
    	{
	        for (int e = row_ptr[u]; e < row_ptr[u + 1]; e++) {
	            
	            v = col_ind[e];
	            w = weights[e];
	            dv = distance[v].x;
	            add = du + w;

	            if (add < dv && du != INT_MAX)
	            {
	            	atomicMin_block(&distance[v].x,add);

	            	if(distance[v].y != *iter)
	            	{
		            	atomicMax_block(&distance[v].y,*iter);
		            	index = atomicAdd_block(nextSize, 1);
		            	//printf("nextSize: %i\n", nextSize);
						nextQueue[index] = v;
					}
	            }
	        }
	    }
    }
}



__global__ void initialize(int2 *distance, int *nv, int *max_distance)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;

	for (int k = i; k < *nv; k += skip) {

		distance[k].x = *max_distance; //instead of INT_MAX, calculated possible max_distance
		distance[k].y = 1; // dont update this in relaxinit. update here. no need for mem copy
	}

	__syncthreads();

}


__global__ void cudasubset_of_vertices(int *size, float *percentage, int *queue, int *subset_queue, unsigned int *random_ints)
{
	int i = threadIdx.x + (blockDim.x * blockIdx.x);

	if (i < *size)
	{
		subset_queue[i] = (queue)[int(random_ints[i]%(*size))];
	}

}