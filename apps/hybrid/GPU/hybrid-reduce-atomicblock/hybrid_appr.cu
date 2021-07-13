
extern "C" {
#include "hybrid.cuh"
}

#include "../../../include/utils/dijkstra_kernel.cuh"

extern "C"


float apprshybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	        const int nv, const int ne, int source, int iter_num, int neg_edge_count)
{
	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_distance, *d_visited, *d_nv, *d_ne,
	    *d_max_distance;
	unsigned int *d_temp_distance;

	
	// Initialize CPU variables
	int max_distance = INT_MAX;
	*distance = (int*)malloc(nv*sizeof(int)); 
	int *visited = (int*)calloc(nv, sizeof(int));
	unsigned int *temp_distance = (unsigned int*)malloc(nv*sizeof(unsigned int));


	// Allocate device
	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_distance, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_visited, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_ne, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_temp_distance, nv*sizeof(unsigned int)));
	cudaCheck(cudaMalloc((void **)&d_max_distance, sizeof(int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_distance, &max_distance, sizeof(int), cudaMemcpyHostToDevice));

	initialize<<<(nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_distance, d_temp_distance, d_visited, d_nv, d_max_distance);

	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(temp_distance, d_temp_distance, nv*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(visited, d_visited, nv*sizeof(int), cudaMemcpyDeviceToHost));

	(*distance)[source] = 0;
	temp_distance[source] = 0;
	visited[source] = 1;

	cudaCheck(cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_temp_distance, temp_distance, nv*sizeof(unsigned int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice));

	int k = neg_edge_count;
	int r = 0;
	printf("neg_edge_count: %i\n", neg_edge_count);

	float elapsed;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));
        

	while(r != (k+2))
  {
      for (int i = 0; i < nv; i++)
		
				visited[i] = 0;

		visited[source] = 1;
		cudaCheck(cudaMemcpy(d_visited, visited, (nv)*sizeof(int), cudaMemcpyHostToDevice));


		int count = 0;

		while(!visitedEmpty(visited, nv) && count < iter_num)
	  {
	        // execute the kernel
	        
	        CUDA_SSSP_KERNEL1_Block<<< (nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK >>>
	                           ( d_row_ptr, d_col_ind, d_weights, d_visited, d_distance, d_temp_distance,
	                             nv, ne );
	       
	        CUDA_SSSP_KERNEL2<<< (nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK >>>
	                           ( d_row_ptr, d_col_ind, d_weights, d_visited, d_distance, d_temp_distance );
	        

	        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * nv, cudaMemcpyDeviceToHost ));

	        count++;
	  }
	  r++;

	}

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

	//Copy outputs to host
	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));


	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_distance));

	printf("GPU SHYBRID time (ms): %f\n", elapsed);

	return elapsed;
}
