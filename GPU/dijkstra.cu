
extern "C" {
#include "dijkstra.cuh"
}

#include <stdio.h>
#include <limits.h>
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

__global__  void CUDA_SSSP_KERNEL1( int *row_ptr, int *col_ind, int *weights,
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
            	atomicMin(&temp_distance[nid], newDist);
            }
        }
    }
}


// This is Algorithm 5 in the paper.  

__global__  void CUDA_SSSP_KERNEL2(  int *row_ptr, int *col_ind, int *weights,
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



extern "C"


void sdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	     const int nv, const int ne, int source, int *signals, float **signal_variables,
	     double *time)
{
	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_distance, *d_visited, *d_nv, *d_ne;
	unsigned int *d_temp_distance;

	
	// Initialize CPU variables
	*distance = (int*)malloc(nv*sizeof(int)); 
	int *visited = (int*)calloc(nv, sizeof(int));
	unsigned int *temp_distance = (unsigned int*)malloc(nv*sizeof(unsigned int));

	for (int i = 0; i < nv; i++)
	{
		(*distance)[i] = INT_MAX;
		temp_distance[i] = INT_MAX;
		visited[i] = 0;
	}

	//printf("source: %i\n", source);

	(*distance)[source] = 0;
	visited[source] = 1;


	// Allocate device
	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_distance, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_visited, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_ne, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_temp_distance, nv*sizeof(unsigned int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_temp_distance, temp_distance, nv*sizeof(unsigned int), cudaMemcpyHostToDevice));


	
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));


	int threadnum = (nv > 1024) ? 1024 : nv;

	int round = 0;

    while(!visitedEmpty(visited, nv))
    {
        int gridSize = (nv + 1023) / 1024;  

        // execute the kernel
        CUDA_SSSP_KERNEL1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance,
                                                nv, ne );
       

        CUDA_SSSP_KERNEL2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
                                                d_visited, d_distance, d_temp_distance );
        

        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * nv, cudaMemcpyDeviceToHost ));

        round++;
        (*signal_variables)[1] = round;
    }
	
	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	float elapsed;
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
	
	//Copy outputs to host
	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));


	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_distance));

	printf("GPU SDJ time: %f\n", elapsed/1000);

	*time = elapsed;
}
