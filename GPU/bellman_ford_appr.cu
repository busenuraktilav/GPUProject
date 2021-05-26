
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
#include <curand.h>
#include <omp.h>

#define N_THREADS_PER_BLOCK 128


#define cudaCheck(error) \
	if (error != cudaSuccess) { \
    	printf("Fatal error: %s at %s:%d\n", \
    		cudaGetErrorString(error), \
    		__FILE__, __LINE__); \
    	exit(1); \
 }

__global__ void cudaRelax(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int* nv, int* distance) 
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

__global__ void cudarelaxAtom(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int2* distance, int *iter) 
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



__global__ void cudarelaxAtomicAddBlock(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int2* distance, int *iter) 
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
	            	index = atomicAdd_block(nextSize, 1);
					nextQueue[index] = v;
				}
            }
        }
    }
}


__global__ void cudarelaxAtomicMinBlock(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int2* distance, int *iter) 
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
            	atomicMin_block(&distance[v].x,add);

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


__global__ void cudarelaxAtomicMaxBlock(int *row_ptr, int *col_ind, int *weights, int *queue, int *nextQueue, int size, int* nextSize, int2* distance, int *iter) 
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
	            	atomicMax_block(&distance[v].y,*iter);
	            	index = atomicAdd(nextSize, 1);
					nextQueue[index] = v;
				}
            }
        }
    }
}



__global__ void cudarelaxAtomicMoreEdges(int *row_ptr, int *col_ind, int *weights, int *queue, 
	                                     int *nextQueue, int size, int* nextSize, int2* distance, 
	                                     int *iter, int *min_edges) 
{

    int index, u, v, w, du, dv, add, tid = threadIdx.x + (blockDim.x * blockIdx.x);

    if (tid < size) { 

    	u = queue[tid];
    	du = distance[u].x;

    	if ((row_ptr[u+1] - row_ptr[u]) > *min_edges) // process only the nodes which has more than ... edges 
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

__global__ void cudainitVar(int2 *distance, int *nv, int *max_distance)
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

	int t = (*size) * (*percentage);
	int temp = t;
	if (i == 1)
	{
		printf("temp: %i, size: %i, percentage: %f\n", temp, *size, *percentage);
	}
	                     

	if (i < t)
	{
		subset_queue[i] = (queue)[random_ints[i]%(t)];
		if ((*percentage) == 0.1 )
		{
			printf("subset_queue[%i]: %i , (queue)[random_ints[%i]=%i % (t=%i)]: %i\n", i, subset_queue[i], i, random_ints[i], t, (queue)[random_ints[i]%(t)]);
		}
		
	}

	(*size) = temp;

}


extern "C"

void apprbf(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	        const int nv, const int ne, int source, int *signals, float **signal_variables, 
	        int max_distance, float *time)
{

	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_nv, *d_max_distance;
	int *d_queue, *d_nextQueue, *d_nextSize, *d_iter, *d_subset_queue, *d_min_edges;
	float *d_percentage;
	int2* d_dist;
	

	int2* dist = (int2*)malloc(nv*sizeof(int2));

	
	//CUDA RANDOM GENERATOR
	curandGenerator_t gen;
	unsigned int *d_random_ints;
	cudaCheck(cudaMalloc((void **)&d_random_ints, nv*sizeof(unsigned int)));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); //Create pseudo-random number generator
	    

	//SIGNALS
	int signal_partial_graph_process = signals[4];
	int signal_reduce_execution = signals[3];
	int iter_num = (*signal_variables)[1];
	float *percentage = (float*) malloc (sizeof(float));
	*percentage = (*signal_variables)[2];
	int min_edges = (*signal_variables)[0];
	int signal_atomicMinBlock = signals[5];
	int signal_atomicMaxBlock = signals[6];
	int signal_atomicAddBlock = signals[7];


	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_dist, nv*sizeof(int2)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_max_distance, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_subset_queue, (nv+1)*sizeof(int))); //space might be percentage*size 
	cudaCheck(cudaMalloc((void **)&d_min_edges, sizeof(int))); //for kernel cudarelaxAtomicMoreEdges, min edges

	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_distance, &max_distance, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_min_edges, &min_edges, sizeof(int), cudaMemcpyHostToDevice));


	cudainitVar<<<(nv + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_dist, d_nv, d_max_distance);

	cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));


	float elapsed = 0.0, elapsed0 = 0.0;

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaCheck(cudaEventCreate(&start));
	cudaCheck(cudaEventCreate(&stop));
	cudaCheck(cudaEventRecord(start, 0));

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

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	cudaCheck(cudaEventElapsedTime(&elapsed0, start, stop));


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

	cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

	if (!signal_reduce_execution && !signal_partial_graph_process && !min_edges && !signal_atomicMinBlock && !signal_atomicMaxBlock && !signal_atomicAddBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			//printf("NO APPR\n");

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtom<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter); 

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers

			(*signal_variables)[1] = round;
			round++;

		}
		
		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}


	if (min_edges && !signal_partial_graph_process && !signal_reduce_execution)
	{
		//min edges to process signal is active. only some edges are processed in kernel

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMoreEdges<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter, d_min_edges); 

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
	}

	
	if (signal_partial_graph_process && !min_edges && !signal_reduce_execution)
	{
		// If reduce signal is negative && partial graph processing signal positive

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtom<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));


			(*iter) ++;	
			std::swap(d_queue, d_nextQueue); // swap frontiers
			curandGenerate(gen, d_random_ints, size); //Generate n ints on device

			cudasubset_of_vertices<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_nextSize, d_percentage, d_queue, d_subset_queue, d_random_ints);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
			
			std::swap(d_queue, d_subset_queue);
			printf("size: %i\n", size);		
			size = *nextSize;
			*nextSize = 0;

			round++;

		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}
	
	
	if (signal_reduce_execution && !signal_partial_graph_process && !min_edges)
	{
		//If reduce signal is positive && partial process signal negative

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((round < iter_num+1) ) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtom<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			round++;

		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}


	if (signal_reduce_execution && min_edges && !signal_partial_graph_process)
	{
		//If reduce signal is positive && partial process signal negative

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((round < iter_num+1) ) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMoreEdges<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter, d_min_edges); 

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			round++;

		}
		
		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}

	if (signal_reduce_execution && signal_partial_graph_process && !min_edges)
	{
		//If reduce signal is positive && partial process signal negative

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((round < iter_num+1) ) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtom<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			curandGenerate(gen, d_random_ints, size); //Generate n ints on device

			cudasubset_of_vertices<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_nextSize, d_percentage, d_queue, d_subset_queue, d_random_ints);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			std::swap(d_queue, d_subset_queue);
			printf("size: %i\n", size);	
			size = *nextSize;
			*nextSize = 0;

			round++;
		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}


	if (signal_partial_graph_process && min_edges && !signal_reduce_execution)
	{
		// If reduce signal is negative && partial graph processing signal positive

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMoreEdges<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter, d_min_edges); 
			
			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			curandGenerate(gen, d_random_ints, size); //Generate n ints on device

			cudasubset_of_vertices<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_nextSize, d_percentage, d_queue, d_subset_queue, d_random_ints);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			std::swap(d_queue, d_subset_queue);
			printf("size: %i\n", size);	
			size = *nextSize;
			*nextSize = 0;

			(*signal_variables)[1] = round;
			round++;

		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}


	if (signal_partial_graph_process && min_edges && signal_reduce_execution)
	{
		// If reduce signal is negative && partial graph processing signal positive

		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((round < iter_num+1) ) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMoreEdges<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter, d_min_edges); 
			
			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			curandGenerate(gen, d_random_ints, size); //Generate n ints on device

			cudasubset_of_vertices<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_nextSize, d_percentage, d_queue, d_subset_queue, d_random_ints);

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			std::swap(d_queue, d_subset_queue);
			printf("size: %i\n", size);		
			size = *nextSize;
			*nextSize = 0;
			round++;

		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}

	
	
	if (signal_atomicAddBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();
	
		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicAddBlock<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter); 

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			round++;

		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}

	
	if (signal_atomicMinBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMinBlock<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter); 

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			round++;
		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}


	if (signal_atomicMaxBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//cudaProfilerStart();

		while((size > 0) && (round < nv) && temp < ne) { temp += size;

			cudaCheck(cudaMemcpy(d_iter, iter, sizeof(int), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(d_nextSize, nextSize, sizeof(int), cudaMemcpyHostToDevice));

			cudarelaxAtomicMaxBlock<<<(size + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK, N_THREADS_PER_BLOCK>>>(d_row_ptr, d_col_ind, d_weights, d_queue, d_nextQueue, size, d_nextSize, d_dist, d_iter); 

			cudaCheck(cudaMemcpy(nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));

			(*iter) ++;
			printf("size: %i\n", size);
			size = *nextSize;
			*nextSize = 0;
			std::swap(d_queue, d_nextQueue); // swap frontiers
			round++;
		}

		//cudaProfilerStop();

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		// Copy outputs to host
		cudaCheck(cudaMemcpy(dist, d_dist, nv*sizeof(int2), cudaMemcpyDeviceToHost));
	}

	

	printf("GPU SBF time(ms): %f\n", elapsed+elapsed0);
	*time = elapsed+elapsed0;

	printf("elapsed: %f, elapsed0: %f, add: %f\n", elapsed, elapsed0, elapsed0+elapsed);
	

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
	cudaCheck(cudaFree(d_subset_queue));
	cudaCheck(cudaFree(d_min_edges));
}