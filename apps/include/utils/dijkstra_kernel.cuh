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


// This is Algorithm 4 in the paper

__global__  void CUDA_SSSP_KERNEL1( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
        {
            int nid = col_ind[edge];

            int w = weights[edge];
            int du = distance[tid];
            int newDist = du + w;

            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
            {
            	atomicMin(&temp_distance[nid], newDist);
            }
        }
    }
}


__global__  void CUDA_SSSP_KERNEL1_Block( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
        {
            int nid = col_ind[edge];

            int w = weights[edge];
            int du = distance[tid];
            int newDist = du + w;

            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
            {
            	atomicMin_block(&temp_distance[nid], newDist);
            }
        }
    }
}



// This is Algorithm 5 in the paper.  

__global__  void CUDA_SSSP_KERNEL2(  int *row_ptr, int *col_ind, int *weights,
                                     int *visited, int *distance, unsigned int *temp_distance)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (distance[tid] > temp_distance[tid])
    {
        distance[tid] = temp_distance[tid];
        visited[tid] = 1;
    }

    temp_distance[tid] = distance[tid];
}



__global__  void CUDA_SSSP_KERNEL1_MoreEdges ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *min_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        if ((row_ptr[tid+1] - row_ptr[tid]) > *min_edges) // process only the nodes which has more than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	            	atomicMin(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}


__global__  void CUDA_SSSP_KERNEL1_MoreEdges_Block ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *min_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        if ((row_ptr[tid+1] - row_ptr[tid]) > *min_edges) // process only the nodes which has more than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	            	atomicMin_block(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}



__global__  void CUDA_SSSP_KERNEL1_LessEdges ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *max_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        if ((row_ptr[tid+1] - row_ptr[tid]) < *max_edges) // process only the nodes which has less than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	            	atomicMin(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}


__global__  void CUDA_SSSP_KERNEL1_LessEdges_Block ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *max_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;

        if ((row_ptr[tid+1] - row_ptr[tid]) < *max_edges) // process only the nodes which has less than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	            	atomicMin_block(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}



__global__  void CUDA_SSSP_KERNEL1_MoreLessEdges ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *min_edges, int *max_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;
        int edge_num = row_ptr[tid+1] - row_ptr[tid];

        if ((edge_num > *min_edges) && (edge_num < *max_edges)) // process only the nodes which has more than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            //if (temp_distance[nid] > (distance[tid] + weights[edge]))
	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	                //temp_distance[nid] = (distance[tid] + weights[edge]);
	                //printf("temp_distance[%i]: %i\n", nid, temp_distance[nid]);
	            	atomicMin(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}


__global__  void CUDA_SSSP_KERNEL1_MoreLessEdges_Block ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *min_edges, int *max_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if ( visited[tid] != 0 )
    {
        visited[tid] = 0;
        int edge_num = row_ptr[tid+1] - row_ptr[tid];

        if ((edge_num > *min_edges) && (edge_num < *max_edges)) // process only the nodes which has more than ... edges 
        {
	        for(int edge = row_ptr[tid]; edge < row_ptr[tid + 1]; edge++)
	        {
	            int nid = col_ind[edge];

	            int w = weights[edge];
	            int du = distance[tid];
	            int newDist = du + w;

	            //if (temp_distance[nid] > (distance[tid] + weights[edge]))
	            if ((temp_distance[nid] > newDist) && (du != INT_MAX))
	            {
	                //temp_distance[nid] = (distance[tid] + weights[edge]);
	                //printf("temp_distance[%i]: %i\n", nid, temp_distance[nid]);
	            	atomicMin_block(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}


__global__ void initialize(int *distance, unsigned int *temp_distance, int *visited, 
	                             int *nv, int *max_distance)
{
	int bdim = blockDim.x, gdim = gridDim.x, bid = blockIdx.x, tid = threadIdx.x;
	int i = bdim * bid + tid;
	int skip = bdim * gdim;

	for (int k = i; k < *nv; k += skip) 
	{
		distance[k] = *max_distance; //instead of INT_MAX, calculated possible max_distance
		temp_distance[k] = *max_distance;
		visited[k] = 0;
	}

	__syncthreads();

}


// Tells if the algorithm needs to continue running or not.

bool visitedEmpty(int *visited, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (visited[i] == 1)

            return false;
    }

    return true;
}


int findCeil(int arr[], int r, int l, int h)
{
    int mid;
    while (l < h)
    {
         mid = l + ((h - l) >> 1);  // Same as mid = (l+h)/2
        (r > arr[mid]) ? (l = mid + 1) : (h = mid);
    }
    return (arr[l] >= r) ? l : -1;
}
 

int myRand(int arr[], int freq[], int n)
{
    int prefix[n], i;
    prefix[0] = freq[0];
    for (i = 1; i < n; ++i)
        prefix[i] = prefix[i - 1] + freq[i];
 
    int r = (rand() % prefix[n - 1]) + 1;

    int indexc = findCeil(prefix, r, 0, n - 1);
    return arr[indexc];
}


bool visitedEmptyRandom(int *visited, int count, int arr[], int freq[], int n)
{
	srand(time(NULL));
	bool flag = 0;

    for(int i = 0; i < count; i++)
    {
        if (visited[i] == 1)
        {
        	flag = 1;

        	if (myRand(arr, freq, n))
        	{
        		visited[i] = 0;
        	}
        }    
    }

    if (flag)
    	return false;
    
    else
    	return true;
}