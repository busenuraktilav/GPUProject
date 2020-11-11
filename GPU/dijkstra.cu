
extern "C" {
#include "dijkstra.cuh"
}

#include <cuda.h>
#include <cuda_profiler_api.h>

#define INF 9999
#define N_THREADS_PER_BLOCK 1024


#define cudaCheck(error) \
	if (error != cudaSuccess) { \
    	printf("Fatal error: %s at %s:%d\n", \
    		cudaGetErrorString(error), \
    		__FILE__, __LINE__); \
    	exit(1); \
 }


__device__ int cudamind(int *dist, int *sptSet, int V)
{
	int min = INT_MAX;
	int min_index;

	for (int v = 0; v < V; v++)

		if (sptSet[v] == 0 && dist[v] <= min)
		{
			min = dist[v];
			min_index = v;
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

		visited[u] = 1;

		if (i < e)
		{
			int tempDistance = distance[u] + weights[(row_ptr)[u]+i];

			if(tempDistance < distance[(col_ind)[(row_ptr)[u]+i]])
			{
				distance[col_ind[(row_ptr)[u]+i]] = tempDistance;
				previous[col_ind[(row_ptr)[u]+i]] = u;
			}
		}
	}
	
}



extern "C"

void sdj(const int *row_ptr, const int *col_ind, const int *weights, int *distance, int *previous, const int nv, const int ne, int source, int *visited)
{
	// Initialize GPU variables
	int *d_row_ptr;
	int *d_col_ind;
	int *d_weights;
	int *d_distance;
	int *d_previous;
	int *d_visited;
	int *d_nv;
	int *d_ne;
	
	// Initialize CPU variables
	distance[source] = 0;


	//int snv = (nv+1)*sizeof(int);
	//int sne = (ne+1)*sizeof(int);

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
	cudaMemcpy(d_distance, distance, nv*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice);


	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	cudasdj<<<(ne+1023)/1024, nv>>>(d_row_ptr, d_col_ind, d_weights, d_distance, d_previous, nv, ne, d_visited);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	

	cudaMemcpy(distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost);


	// Deallocation
	cudaFree(d_row_ptr);
	cudaFree(d_col_ind);
	cudaFree(d_weights);
	cudaFree(d_distance);
	cudaFree(d_previous);

	printf("GPU SDJ time: %f\n", elapsed/1000);
}