
extern "C" {
#include "bellman_ford.cuh"
}

#include "../../../include/utils/bellman_ford_kernel.cuh"

extern "C"

float apprbf(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	        const int nv, const int ne, int source, int min_edges, int max_edges)
{
	int max_distance = INT_MAX;

	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_nv, *d_max_distance;
	int *d_queue, *d_nextQueue, *d_nextSize, *d_iter, *d_min_edges, *d_max_edges;
	int2* d_dist;
	

	int2* dist = (int2*)malloc(nv*sizeof(int2));

	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_dist, nv*sizeof(int2)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_max_distance, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_min_edges, sizeof(int))); //for kernel cudarelaxAtomicMoreEdges, min edges
	cudaCheck(cudaMalloc((void **)&d_max_edges, sizeof(int)));

	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_distance, &max_distance, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_min_edges, &min_edges, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_edges, &max_edges, sizeof(int), cudaMemcpyHostToDevice));

	initialize<<<(nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_dist, d_nv, d_max_distance);

	cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));

	float elapsed = 0.0, elapsed0 = 0.0;

	cudaEvent_t start0;
	cudaEvent_t stop0;
	cudaCheck(cudaEventCreate(&start0));
	cudaCheck(cudaEventCreate(&stop0));
	cudaCheck(cudaEventRecord(start0, 0));

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

	cudaCheck(cudaEventRecord(stop0, 0));
	cudaCheck(cudaEventSynchronize(stop0));
	cudaCheck(cudaEventElapsedTime(&elapsed0, start0, stop0));


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

	cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

	//min edges to process signal is active. only some edges are processed in kernel

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

	//cudaProfilerStart();

	int *d_size;
	cudaCheck(cudaMalloc((void **)&d_size, sizeof(int)));

	while((size > 0)) { 

		cudaCheck(cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

		cudarelaxAtomicMoreLessEdgesBlock<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>
		                           (d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, d_size, d_nextSize, d_dist, 
		                           	d_iter, d_min_edges, d_max_edges); 

		cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

		(*iter) ++;
		printf("size: %i\n", size);
		size = *nextSize;
		*nextSize = 0;
		std::swap(d_queue, d_nextQueue); // swap frontiers
	}

	//cudaProfilerStop();

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

	// Copy outputs to host
	cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));

	// check for negative cycles
	int neg_cycle = false;

	for (int v2 = 0; v2 < nv; v2++)
	{
		for (int e = (row_ptr)[v2]; e < (row_ptr)[v2+1]; e++)
		{
			int tempDistance = dist[v2].x + (weights)[e];
			int neigh = (col_ind)[e];

			if ((tempDistance < dist[neigh].x) && (dist[v2].x != INT_MAX))
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
	cudaCheck(cudaFree(d_nv));
	cudaCheck(cudaFree(d_max_distance));
	cudaCheck(cudaFree(d_queue));
	cudaCheck(cudaFree(d_nextQueue));
	cudaCheck(cudaFree(d_nextSize));
	cudaCheck(cudaFree(d_iter));
	cudaCheck(cudaFree(d_min_edges));
	cudaCheck(cudaFree(d_max_edges));


	printf("GPU SBF time(ms): %f\n", elapsed+elapsed0);

	return (elapsed+elapsed0);
	
}