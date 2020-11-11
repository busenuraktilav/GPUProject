
extern "C" {
#include "bellman_ford.cuh"
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

__global__ void cudasbf(int *row_ptr, int *col_ind, int *row_ind, int *weights, int *distance, int *previous, int nv, int ne)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for(int v = 0; v < nv; v++)
	{
		if(tid < ne)
		{
			int tempdistance = distance[(row_ind)[tid]] + weights[tid];

			if(tempdistance < distance[(col_ind)[tid]])
			{
				distance[(col_ind)[tid]] = tempdistance;
				previous[(col_ind)[tid]] = row_ind[tid];
			}
		}
	}
}


extern "C"

void sbf(const int *row_ptr, const int *col_ind, const int *row_ind, const int *weights, int *distance, int *previous, const int nv, const int ne, int source)
{
	// Initialize GPU variables
	int *d_row_ptr;
	int *d_col_ind;
	int *d_row_ind;
	int *d_weights;
	int *d_distance;
	int *d_previous;
	int *d_nv;
	int *d_ne;
	
	// Initialize CPU variables
	distance[source] = 0;


	//int snv = (nv+1)*sizeof(int);
	//int sne = (ne+1)*sizeof(int);

	// Allocate device
	cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int));
	cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int));
	cudaMalloc((void **)&d_row_ind, (ne+1)*sizeof(int));
	cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int));
	cudaMalloc((void **)&d_distance, nv*sizeof(int));
	cudaMalloc((void **)&d_previous, nv*sizeof(int));
	cudaMalloc((void **)&d_nv, sizeof(int));
	cudaMalloc((void **)&d_ne, sizeof(int));

	//Copy inputs to device
	cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row_ind, row_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_distance, distance, nv*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice);


	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudasbf<<<(ne+1023)/1024, 1024>>>(d_row_ptr, d_col_ind, d_row_ind, d_weights, d_distance, d_previous, nv, ne);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	// Copy outputs to host
	cudaMemcpy(distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost);


	// check for negative cycles
	for(int e = 0; e < ne; e++)
	{
		if(distance[row_ind[e]] + weights[e] < distance[(col_ind)[e]])
			printf("Error: negative cycle exists\n");
	}


	// Deallocation
	cudaFree(d_row_ptr);
	cudaFree(d_col_ind);
	cudaFree(d_row_ind);
	cudaFree(d_weights);
	cudaFree(d_distance);
	cudaFree(d_previous);

	printf("GPU SBF time: %f\n", elapsed);
}