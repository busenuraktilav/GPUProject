
extern "C" {
#include "dijkstra.cuh"
}

#include <stdio.h>
#include <limits.h>
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


__device__ int cudamind(int *dist, int *visited, int V)
{
	int min = INT_MAX;
	int min_index;


	for (int v = 0; v < V; v++)
	{

		//printf("dist[%i]: %i, visited[%i]: %i\n", v, dist[v], v, visited[v]);

		if (visited[v] == 0 && dist[v] <= min)
		{
			min = dist[v];
			min_index = v;
		}
	}

	return min_index;
}


__global__ void cudasdj(int *row_ptr, int *col_ind, int *weights, int *distance, int *previous, int nv, int ne, int *visited)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	for (int count = 0; count < nv; count++)
	{

		int u = cudamind(distance, visited, nv);
		int e = row_ptr[u+1] - row_ptr[u];

		//printf("%i, %i\n", u, e);

		visited[u] = 1;		

		if (i < e)
		{
			int tempDistance = distance[u] + weights[(row_ptr)[u]+i];

			//printf("%i - distance[%i]: %i, weights[(row_ptr)[u]+i]: %i, tempDistance: %i, distance[(col_ind)[(row_ptr)[u]+i]]: %i\n", i, u, distance[u], weights[(row_ptr)[u]+i], tempDistance, distance[(col_ind)[(row_ptr)[u]+i]]);

			if(tempDistance < distance[(col_ind)[(row_ptr)[u]+i]] && distance[u] != INT_MAX)
			{
				distance[col_ind[(row_ptr)[u]+i]] = tempDistance;
				previous[col_ind[(row_ptr)[u]+i]] = u;
			}
		}
	}
	
}



extern "C"

void sdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source)
{
	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_distance, *d_previous, *d_visited, *d_nv, *d_ne;

	
	// Initialize CPU variables
	*distance = (int*)malloc(nv*sizeof(int)); 
	*previous = (int*)malloc(nv*sizeof(int));
	int *visited = (int*)calloc(nv, sizeof(int));

	for (int i = 0; i < nv; i++)
	{
		(*distance)[i] = INT_MAX;
		(*previous)[i] = -1;
	}

	(*distance)[source] = 0;


	// Allocate device
	cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int));
	cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int));
	cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int));
	cudaMalloc((void **)&d_distance, nv*sizeof(int));
	cudaMalloc((void **)&d_previous, nv*sizeof(int));
	cudaMalloc((void **)&d_visited, nv*sizeof(int));
	cudaMalloc((void **)&d_nv, sizeof(int));
	cudaMalloc((void **)&d_ne, sizeof(int));

	//Copy inputs to device
	cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice);


	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int threadnum = (nv > 1024) ? 1024 : nv;

	cudasdj<<<10, threadnum>>>(d_row_ptr, d_col_ind, d_weights, d_distance, d_previous, nv, ne, d_visited);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	
	//Copy outputs to host
	cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*previous, d_previous, nv*sizeof(int), cudaMemcpyDeviceToHost);


	// Deallocation
	cudaFree(d_row_ptr);
	cudaFree(d_col_ind);
	cudaFree(d_weights);
	cudaFree(d_distance);
	cudaFree(d_previous);

	printf("GPU SDJ time: %f\n", elapsed/1000);
}