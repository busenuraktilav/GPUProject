
extern "C" {
#include "dijkstra.cuh"
}

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_profiler_api.h>


#define cudaCheck(error) \
	if (error != cudaSuccess) { \
    	printf("Fatal error: %s at %s:%d\n", \
    		cudaGetErrorString(error), \
    		__FILE__, __LINE__); \
    	exit(1); \
 }


// This is Algorithm 4 in the paper

__global__  void cudaSSSPKernel1 ( int *row_ptr, int *col_ind, int *weights,
                                     int *visited, int *distance, unsigned int *temp_distance,
                                     int nv, int ne )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
    	//printf("inside kernel1: %i\n", tid);
        visited[tid] = 0;

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
            	atomicExch(&temp_distance[nid], newDist);
            }
        }
    }
}


// This is Algorithm 5 in the paper.  

__global__  void cudaSSSPKernel2 ( int *row_ptr, int *col_ind, int *weights,
                                     int *visited, int *distance, unsigned int *temp_distance)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("INSIDE KERNEL2\n");

    if (distance[tid] > temp_distance[tid])
    {
        distance[tid] = temp_distance[tid];
        visited[tid] = 1;
        //printf("kernel2 -> distance[%i]: %i\n", tid, distance[tid]);
    }

    temp_distance[tid] = distance[tid];
}



__global__  void cudaSSSPKernel1MoreEdges ( int *row_ptr, int *col_ind, int *weights,
                                    int *visited, int *distance, unsigned int *temp_distance,
                                    int nv, int ne, int *min_edges )
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( visited[tid] != 0 )
    {
    	//printf("inside kernel1: %i\n", tid);
        visited[tid] = 0;

        if ((row_ptr[tid+1] - row_ptr[tid]) > *min_edges) // process only the nodes which has more than ... edges 
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
	            	atomicExch(&temp_distance[nid], newDist);
	            }
	        }
	    }
    }
}

__global__ void cudaInitDistance(int *distance, unsigned int *temp_distance, int *visited, int *nv, int *max_distance)
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

bool visitEmpty(int *visited, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (visited[i] == 1)

            return false;
    }

    return true;
}



extern "C"


void apprsdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source, float **appr_vals, int max_distance, float *time)
{
	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_distance, *d_previous, *d_visited, *d_nv, *d_ne,
	    *d_max_distance, *d_min_edges;
	unsigned int *d_temp_distance;

	
	// Initialize CPU variables
	*distance = (int*)malloc(nv*sizeof(int)); 
	*previous = (int*)malloc(nv*sizeof(int));
	int *visited = (int*)calloc(nv, sizeof(int));
	unsigned int *temp_distance = (unsigned int*)malloc(nv*sizeof(unsigned int));


	//SIGNALSs
	int signal_partial_graph_process = (*appr_vals)[0];
	int signal_reduce_execution = (*appr_vals)[1];
	int iter_num = (*appr_vals)[2];
	float *percentage = (float*)malloc(nv*sizeof(float));
	*percentage = (*appr_vals)[3];
	int min_edges = (*appr_vals)[4];


	// Allocate device
	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_distance, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_previous, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_visited, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_ne, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_temp_distance, nv*sizeof(unsigned int)));
	cudaCheck(cudaMalloc((void **)&d_max_distance, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_min_edges, sizeof(int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	//cudaCheck(cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice));
	//cudaCheck(cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice));
	//cudaCheck(cudaMemcpy(d_temp_distance, temp_distance, nv*sizeof(unsigned int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_distance, &max_distance, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_min_edges, &min_edges, sizeof(int), cudaMemcpyHostToDevice));


	cudaEvent_t start;
	cudaEvent_t stop;

	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	int threadnum = 128;
	int gridSize = (nv + threadnum - 1) / threadnum;  


	cudaInitDistance<<<gridSize, threadnum>>>(d_distance, d_temp_distance, d_visited, d_nv, d_max_distance);

	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(temp_distance, d_temp_distance, nv*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(visited, d_visited, nv*sizeof(int), cudaMemcpyDeviceToHost));

	(*distance)[source] = 0;
	visited[source] = 1;

	cudaCheck(cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice));

	int count = 0;


	// no approximation. Both signals are negative
    while(!visitEmpty(visited, nv) && !signal_reduce_execution && !signal_partial_graph_process && !min_edges)
    {
        // execute the kernel
        cudaSSSPKernel1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance,
                                                nv, ne );
       

        cudaSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance );
        

        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * nv, cudaMemcpyDeviceToHost ));

        (count)++;
        //printf("count: %i\n", *count);
    }

    //min edges to process signal is active. only some edges are processed in kernel
    while(!visitEmpty(visited, nv) && !signal_reduce_execution && !signal_partial_graph_process && min_edges)
    {
        // execute the kernel
        cudaSSSPKernel1MoreEdges<<< gridSize, threadnum >>> ( d_row_ptr, d_col_ind, d_weights,
                                   							  d_visited, d_distance, d_temp_distance,
                                   							  nv, ne, d_min_edges );
        //cudaSSSPKernel1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
        //                                        d_visited, d_distance, d_temp_distance,
        //                                        nv, ne );
       

        cudaSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance );
        

        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * nv, cudaMemcpyDeviceToHost ));

        (count)++;
        //printf("count: %i\n", *count);
    }


    while(!visitEmpty(visited, nv) && count <= iter_num && signal_reduce_execution && !signal_partial_graph_process && !min_edges)
    {
        // execute the kernel
        cudaSSSPKernel1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance,
                                                nv, ne );
       

        cudaSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance );
        

        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * nv, cudaMemcpyDeviceToHost ));

        (count)++;
        //printf("count: %i\n", *count);
    }


    //Copy outputs to host
	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	
	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	float elapsed;
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
	

	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_distance));
	cudaCheck(cudaFree(d_previous));

	printf("GPU SDJ time (ms): %f\n", elapsed);

	*time = elapsed;
}
