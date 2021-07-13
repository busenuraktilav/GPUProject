#include "cpu_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>


float relative_error (int **arr1, int **arr2, int size)
{
	float err = 0.0;
	float actual_val = 0.0;
	float error = 0.0;


	for (int i = 0; i < size; i++)
	{
		if(((*arr1)[i] == INT_MAX && (*arr2)[i]!= INT_MAX)  || ((*arr2)[i] == INT_MAX && (*arr1)[i] != INT_MAX))
		{
			error += 1;
		}
		
		else 
		{
			err = abs((*arr1)[i] - (*arr2)[i]);
			actual_val = (*arr1)[i];
			
			if(!isnan(err/actual_val) && actual_val){
				error += err/actual_val;
			}
			else if (!isnan(err/actual_val) && !actual_val){
				error += (err/abs((*arr1)[i] + (*arr2)[i]))*2;
			}
		}
	}

	return (error/size)*100;
}


void init_zero (int **arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		(*arr)[i] = 0;
	}
}


//Technique 4: round weights to the nearest power of 2
void approximate_attributes (int **weights, int ne, int *max_weight, int *min_weight)
{
	*max_weight = pow(2,round(log2(*max_weight)));
	*min_weight = pow(2,round(log2(*min_weight)));

	int k = log2(*max_weight) - log2(*min_weight);

	//int max_distance = level * (*max_weight);

	for (int i = 0; i < ne; i++)
	{
		(*weights)[i] = pow(2,round(log2((*weights)[i])));
	}
}


int sync_bfs (int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source)
{
    memset(level_arr, -1, nv * sizeof(int));
    int level = 0;
    level_arr[source] = level;
    bool improvement = true;

    while (improvement)
    {
        improvement = false;
        for (int v = 0; v < nv; ++v)
        {
            if (level_arr[v] == level)
            {
                for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; ++edge)
                {
                    int adj = col_ind[edge];
                    if (level_arr[adj] == -1)
                    {
                        improvement = true;
                        level_arr[adj] = level + 1;
                    }
                }
            }
        }
        level++;
    }
    return level;
}

/*
int write_performance_results(const char *filename, int nv, int ne, int iter_num, int max_degree, 
	                          int min_edge, float percentage, bool signal_originalDistance, 
	                          bool signal_kernelMinEdge, bool signal_appr_attr, bool signal_reduce_execution, 
	                          bool signal_partial_graph_process, float error, float time)
*/
int write_time_results (const char *filename, float time)
{
	FILE *fp = fopen(filename, "at");

	if(fp == NULL)
		return -1;

	fprintf(fp, "%f\n", time);

	fclose(fp);

	return 1;
}


float find_max_time (float *time, int repeat)
{
	float max = 0.0;

	if (repeat < 3)
		return 0;

	for (int i = 0; i < repeat; ++i)
	{
		if (max < time[i])
			max = time[i];
	}

	return max;
}


float find_min_time (float *time, int repeat, float max)
{
	float min = max;

	for (int i = 0; i < repeat; ++i)
	{
		if (min > time[i])
			min = time[i];
	}

	return min;
}


int count_lines (const char *filename)
{
	FILE *fp = fopen(filename, "r");
	char line[1025];

	int d, count = 0;

	if (fp == NULL)
	{
		printf("File could not be found!");
		return -1;
	}

	while (fgets(line, 1025, fp) != NULL)
	{
		sscanf(line, "%d", &d);
		count++;
	}

	fclose(fp);

	return count;
}


int read_time (const char* filename, float **time, int *repeat)
{
	FILE *fp = fopen(filename, "r");
	char line[1025];

	*repeat = count_lines(filename);
	*time = (float *)malloc((*repeat)*sizeof(float));

	int i;
	float d;

	if(fp == NULL)
	{
		printf("File could not be found!");
		return -1;
	}


	while (fgets(line, 1025, fp) != NULL)
	{
		sscanf(line, "%f", &d);

		(*time)[i++] = d;

		//printf("d: %f, time[%i]: %f\n", d, i, (*time)[i] );
	}

	fclose(fp);

	return 1;
}


float find_avg_time (const char *filename)
{
	float *time, avg = 0;
	int repeat;

	if (read_time(filename, &time, &repeat) != 1)
	{
		printf("Something went terribly wrong in read_time file!");
		return -1;
	}

	if (repeat < 3)
	{
		return time[0];
	}

	for (int i = 0; i < repeat; ++i)
	{
		avg += time[i];
		//printf("time[%i]: %f\n", i, time[i]);
	}

	float max = find_max_time(time, repeat);
	float min = find_min_time(time, repeat, max);

	//printf("avg: %f, max: %f, min: %f\n", avg, max, min);

	avg -= max;
	avg -= min;

	avg = (avg/(repeat-2));
	//printf("avg: %f, max: %f, min: %f, repeat: %i\n", avg, max, min, repeat);

	return avg;
}


int write_performance_results(const char *perf_file, const char *time_file, const char *error_file, 
	                          int nv, int ne,  int max_degree, int *signals, int min_edges, 
	                          int max_edges, int iter_num, float percentage)
{
	FILE *fptr;
	float time = 0, error = 0;
	
	time = find_avg_time(time_file);

	if (( fptr = fopen( error_file, "r" )))
	{
		error = find_avg_time(error_file);
	}

	FILE *fp = fopen(perf_file, "at");

	if(fp == NULL)
		return -1;
	
	fprintf(fp, "%d,%d,%d,%d,%d,%d,%.2f,%d,%d,%d,%d,%d,%d,%.4f,%.4f\n", nv, ne, iter_num, max_degree, 
		                                            min_edges, max_edges, percentage, signals[0], signals[1], 
		                                            signals[2], signals[3], signals[4], signals[5], error, time);
	


	fclose(fp);

	return 1;
}

int cmpfuncmin (const void * a, const void * b)
{
	return ( *(int*)a - *(int*)b );
}

int cmpfuncmax (const void * a, const void * b)
{
	return ( *(int*)b - *(int*)a );
}

int eliminate_repeated_elements_from_array (const int *arr, int size, int new_size, int min_edge)
{
	int count = 0;

	int new_arr[new_size];

	for (int i = 1; i < size; ++i)
	{
		if (arr[i-1] != arr[i] && arr[i] > arr[i-1])
		{
			new_arr[count] = arr[i];
			count++;

			if (count == new_size) break;
		}
	}

	return new_arr[min_edge-1];

}


int min_edge_to_process (const int *row_ptr, int nv, int min_edge)
{
	int *arr = (int *) malloc (nv * sizeof(int));

	for (int i = 1; i <= nv ; ++i)
	{
		arr[i-1] = row_ptr[i] - row_ptr[i - 1];
	}

	qsort(arr, nv, sizeof(int), cmpfuncmin);

	int start = 0;
	
	for (int i = 0; i < nv; ++i)
	{
		if (arr[i] != 0)
		{
			start = i;
			break;
		}
	}

	int nchunk = 10;
	int chunk = ((nv - start) / nchunk) + 1;
	int j = start;
	int size = nchunk*3;

	printf("chunk in min: %i\n", chunk);

	int *indices = (int *) malloc (size * sizeof(int));

	for (int i = 0; i < nchunk; ++i)
	{
		indices[i] = arr[j];

		if (i == nchunk - 1 && j < nv)
		{
			int chunk2 = chunk / 10;
			int z = chunk2;

			for (int k = 1; k < 10; ++k)
			{
				indices[i+k] = arr[j+z];
				z += chunk2;

			}
		}
		
		j+=chunk;
 	}

 	int new_size = nchunk*2;
 	 	
 	int r = eliminate_repeated_elements_from_array (indices, size, new_size, min_edge);

 	return r;

}


int max_edge_to_process (const int *row_ptr, int nv, int max_edge)
{
	int *arr = (int *) malloc (nv * sizeof(int));

	for (int i = 1; i <= nv ; ++i)
	{
		arr[i-1] = row_ptr[i] - row_ptr[i - 1];
	}

	qsort(arr, nv, sizeof(int), cmpfuncmax);

	int start = 0;
	
	for (int i = 0; i < nv; ++i)
	{
		if (arr[i] != 0)
		{
			start = i;
			break;
		}
	}

	int nchunk = 10;
	int chunk = (((nv - start) / 200))/nchunk;
	printf("chunk in max: %i\n", chunk);
	int j = 0;

	int *indices = (int *) malloc (nchunk * sizeof(int));

	for (int i = 0; i < nchunk; ++i)
	{
		indices[i] = arr[j+1];
		j+=chunk;
 	}

 	return indices[max_edge-1];

 }



int select_start_vertex(int *row_ptr, int nv, int *max_degree)
{
	int deg, start = 0;

	// Select the starting vertex (vertex that contains the max num of edge)
	for (int i = 0; i < nv; ++i)
	{
		deg = row_ptr[i + 1] - row_ptr[i];
		if (deg > (*max_degree))
		{
			start = i;
			(*max_degree) = deg;
		}
	}

	return start;
}

