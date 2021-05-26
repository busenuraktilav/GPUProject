
extern "C" {
#include "hybrid.cuh"
//#include "../graphio/graphio.h"
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


__global__  void cudaHSSSPKernel1 ( int *row_ptr, int *col_ind, int *weights,
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


__global__  void cudaHSSSPKernel1AtomicExchBlock ( int *row_ptr, int *col_ind, int *weights,
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
            	atomicMin_block(&temp_distance[nid], newDist);
            }
        }
    }
}



  

__global__  void cudaHSSSPKernel2 ( int *row_ptr, int *col_ind, int *weights,
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


__global__  void cudaHSSSPKernel1MoreEdges ( int *row_ptr, int *col_ind, int *weights,
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

__global__ void cudainitDistance(int *distance, unsigned int *temp_distance, int *visited, 
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


bool visitempty(int *visited, int count)
{
    for(int i = 0; i < count; i++ )
    {
        if (visited[i] == 1)

            return false;
    }

    return true;
}



extern "C"


void apprshybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	         const int nv, const int ne, int source, int *signals, float **signal_variables, 
	         int max_distance, int neg_edge_count, float *time)
{
	// Initialize GPU variables
	int *d_row_ptr, *d_col_ind, *d_weights, *d_distance, *d_previous, *d_visited, *d_nv, *d_ne,
	    *d_max_distance, *d_min_edges;
	unsigned int *d_temp_distance;

	
	// Initialize CPU variables
	*distance = (int*)malloc(nv*sizeof(int)); 
	int *visited = (int*)calloc(nv, sizeof(int));
	int *temp_distance = (int*)malloc(nv*sizeof(int));

	
	//SIGNALSs
	int signal_partial_graph_process = signals[4];
	int signal_reduce_execution = signals[3];
	int iter_num = (*signal_variables)[1];
	float *percentage = (float*) malloc (sizeof(float));
	*percentage = (*signal_variables)[2];
	int min_edges = (*signal_variables)[0];
	int signal_atomicExchBlock = signals[5];


	// Allocate device
	cudaCheck(cudaMalloc((void **)&d_row_ptr, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_col_ind, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_weights, (ne+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_distance, nv*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_visited, (nv+1)*sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_nv, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_ne, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_temp_distance, nv*sizeof(unsigned int)));
	cudaCheck(cudaMalloc((void **)&d_max_distance, sizeof(int)));
	cudaCheck(cudaMalloc((void **)&d_min_edges, sizeof(int)));

	//Copy inputs to device
	cudaCheck(cudaMemcpy(d_row_ptr, row_ptr, (nv+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_col_ind, col_ind, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_weights, weights, (ne+1)*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_nv, &nv, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_ne, &ne, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_max_distance, &max_distance, sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_min_edges, &min_edges, sizeof(int), cudaMemcpyHostToDevice));


	int threadnum = 128;
	int gridSize = (nv + threadnum - 1) / threadnum; 


	cudainitDistance<<<gridSize, threadnum>>>(d_distance, d_temp_distance, d_visited, d_nv, d_max_distance);

	cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(temp_distance, d_temp_distance, nv*sizeof(unsigned int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(visited, d_visited, nv*sizeof(int), cudaMemcpyDeviceToHost));


	(*distance)[source] = 0;
	visited[source] = 1;

	cudaCheck(cudaMemcpy(d_distance, *distance, nv*sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(d_visited, visited, nv*sizeof(int), cudaMemcpyHostToDevice));


	int k = neg_edge_count;

	int count = 0;

	float elapsed;

	printf("neg_edge_count: %i\n", neg_edge_count);


	if (!signal_reduce_execution && !signal_partial_graph_process && !min_edges && !signal_atomicExchBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		// no approximation
		while((count) != (k+2))
	    {
	    	int cnt = 0;

	    	while(!visitempty(visited, nv))
	    	{
		        // execute the kernel
		        cudaHSSSPKernel1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance,
		                                                nv, ne );


		        cudaHSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance );

		        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * (nv+1), cudaMemcpyDeviceToHost ));
	       
	       		cnt++;
	       		
	       		(*signal_variables)[1] = cnt;

	        }
	        (count)++;
	    }

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		//Copy outputs to host
		cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
		

	}


	if (min_edges && !signal_reduce_execution && !signal_atomicExchBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		//min edges to process signal is active. only some edges are processed in kernel
	    while((count) != (k+2))
	    {
	    	 while(!visitempty(visited, nv))
	    	 {
		        // execute the kernel
		        cudaHSSSPKernel1MoreEdges<<< gridSize, threadnum >>> ( d_row_ptr, d_col_ind, d_weights,
	                                   							  d_visited, d_distance, d_temp_distance,
	                                   							  nv, ne, d_min_edges );
		       

		        cudaHSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance );
		        

		        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * (nv+1), cudaMemcpyDeviceToHost ));
		       
	        }
	        (count)++;
	    }

		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		//Copy outputs to host
		cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));
	}

    
	if (signal_reduce_execution && !min_edges && !signal_atomicExchBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		while((count) != (k+2))
	    {
	    	int cnt = 0;

	    	while(!visitempty(visited, nv) && cnt <= iter_num)
	    	{
		        // execute the kernel
		        cudaHSSSPKernel1<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance,
		                                                nv, ne );
		       
		        cudaHSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance );
		        
		        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * (nv+1), cudaMemcpyDeviceToHost ));
		      
		        cnt++;
	        }

	        (count)++;
	    
	    }

	    
		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		//Copy outputs to host
		cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));	

	}


	if (signal_reduce_execution && min_edges && !signal_atomicExchBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		while((count) != (k+2))
	    {

	    	int cnt = 0;

	    	while(!visitempty(visited, nv) && cnt <= iter_num)
	    	{
		        // execute the kernel
		        cudaHSSSPKernel1MoreEdges<<< gridSize, threadnum >>> ( d_row_ptr, d_col_ind, d_weights,
	                                   							  d_visited, d_distance, d_temp_distance,
	                                   							  nv, ne, d_min_edges );
		       

		        cudaHSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance );
		        

		        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * (nv+1), cudaMemcpyDeviceToHost ));
		      
		        cnt++;
	        }

	        (count)++;
	    }

	    
		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		//Copy outputs to host
		cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));	

	}

	
	if (signal_atomicExchBlock)
	{
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaCheck(cudaEventCreate(&start));
		cudaCheck(cudaEventCreate(&stop));
		cudaCheck(cudaEventRecord(start, 0));

		while((count) != (k+2))
	    {
	    	int cnt = 0;

	    	while(!visitempty(visited, nv))
	    	{
		        // execute the kernel
		        cudaHSSSPKernel1AtomicExchBlock<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance,
		                                                nv, ne );
		       

		        cudaHSSSPKernel2<<< gridSize, threadnum >>>( d_row_ptr, d_col_ind, d_weights,
		                                                d_visited, d_distance, d_temp_distance );
		        

		        cudaCheck(cudaMemcpy( visited, d_visited, sizeof(int) * (nv+1), cudaMemcpyDeviceToHost ));
		      
		        cnt++;
	        }
	        (count)++;
	    }

	    
		cudaCheck(cudaEventRecord(stop, 0));
		cudaCheck(cudaEventSynchronize(stop));
		cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));

		//Copy outputs to host
		cudaCheck(cudaMemcpy(*distance, d_distance, nv*sizeof(int), cudaMemcpyDeviceToHost));	

	}
	

	printf("count: %i\n", count);

	// Deallocation
	cudaCheck(cudaFree(d_row_ptr));
	cudaCheck(cudaFree(d_col_ind));
	cudaCheck(cudaFree(d_weights));
	cudaCheck(cudaFree(d_distance));

	printf("GPU SHYBRID time: %f\n", elapsed);

	*time = elapsed;

	
}