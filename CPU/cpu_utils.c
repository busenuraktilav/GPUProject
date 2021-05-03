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


float jaccard_similarity (int *row_ptr, int *col_ind, int v1, int v2, int *conj, int *diff)
{
	int edge_num1 = row_ptr[v1+1] - row_ptr[v1];
	int edge_num2 = row_ptr[v2+1] - row_ptr[v2];

	for (int i = 0; i < edge_num1; ++i)
	{
		for (int j = 0; j < edge_num2; ++j)
		{
			if (col_ind[row_ptr[v1]+i] == col_ind[row_ptr[v2]+j])
				(*conj)++;
			else
				(*diff)++;
		}
	}

	return (*conj)/((*conj)+(*diff));
}


//Technique 3: Approximate graph representation
void appr_graph_rep (int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int similarity_rate)
{
	int diff = 0; 
	int conj = 0;

	for (int v1 = 0; v1 < *nv; v1++)
	{
		for (int v2 = 0; v2 < *nv; v2++)
		{
			int js = jaccard_similarity(*row_ptr, *col_ind, v1, v2, &conj, &diff);

			if ((v1 != v2) && (js < similarity_rate))
			{
				//merge_nodes(row_ptr, col_ind, weights, nv, ne, v1, v2, conj, diff);
			}
		}
	}
}

int write_distance (const char *filename, int *distance, int *iter_num, int *max_degree, int nv)
{
	FILE *fp = fopen(filename, "w");

	if(fp == NULL)
		return -1;


	fprintf(fp, "%d\n", *iter_num);
	fprintf(fp, "%d\n", *max_degree);

	for (int i = 0; i < nv; i++)
	{
		fprintf(fp, "%d\n", distance[i]);
	}

	fclose(fp);

	return 1;
}

int read_distance (const char *filename, int **distance, int *iter_num, int *max_degree, int nv)
{
	FILE *fp = fopen(filename, "r");
	char line[1025];

	*distance = (int *)malloc(nv*sizeof(int));

	int i, d;

	if(fp == NULL)
	{
		printf("File could not be found!");
		return -1;
	}


	fgets(line, 1025, fp);
	sscanf(line, "%d", iter_num);
	fgets(line, 1025, fp);
	sscanf(line, "%d", max_degree);

	for (i = 0; i < nv; ++i)
	{
		fgets(line, 1025, fp);
		sscanf(line, "%d", &d);

		(*distance)[i] = d;
	}

	return 1;
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


int read_time (const char* filename, float **time, int repeat)
{
	FILE *fp = fopen(filename, "r");
	char line[1025];

	*time = (float *)malloc(repeat*sizeof(float));

	int i, d;

	if(fp == NULL)
	{
		printf("File could not be found!");
		return -1;
	}


	for (i = 0; i < repeat; ++i)
	{
		fgets(line, 1025, fp);
		sscanf(line, "%d", &d);

		(*time)[i] = d;
	}

	return 1;
}


float find_avg_time (const char *filename, int repeat)
{
	float *time, avg;
	
	read_time(filename, &time, repeat);

	for (int i = 0; i < repeat; ++i)
	{
		avg += time[i];
	}

	float max = find_max_time(time, repeat);
	float min = find_min_time(time, repeat, max);

	avg -= max;
	avg -= min;

	avg = avg/(repeat-2);

	return avg;
}


int write_performance_results(const char *perf_file, const char *time_file, int nv, int ne, int iter_num, int max_degree, 
							  int min_edge, float percentage, bool signal_originalDistance, bool signal_kernelMinEdge, 
							  bool signal_appr_attr, bool signal_reduce_execution, bool signal_partial_graph_process,
							  bool signal_atomicMinBlock, bool signal_atomicMaxBlock, bool signal_atomicAddBlock, float error)
{
	int repeat = 10;

	float time = find_avg_time(time_file, repeat);

	FILE *fp = fopen(perf_file, "at");

	if(fp == NULL)
		return -1;
	
	fprintf(fp, "%d,%d,%d,%d,%d,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f\n", nv, ne, iter_num, max_degree, min_edge, 
		                                            percentage, signal_originalDistance, signal_kernelMinEdge, 
		                                            signal_appr_attr, signal_reduce_execution, 
		                                            signal_partial_graph_process, signal_atomicMinBlock, 
		                                            signal_atomicMaxBlock, signal_atomicAddBlock, error, time);
	


	fclose(fp);

	return 1;
}

int cmpfunc (const void * a, const void * b)
{
	return ( *(int*)a - *(int*)b );
}

int eliminate_repeated_elements_from_array (const int *arr, int size, int new_size, int min_edge)
{
	int count = 0;

	int new_arr[new_size];

	for (int i = 1; i < size; ++i)
	{
		if (arr[i-1] != arr[i] && arr[i] > arr[i-1])
		{
			(new_arr)[count] = arr[i];
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
		//printf("row_ptr[%i]: %i\n", i, row_ptr[i]);
	}

	qsort(arr, nv, sizeof(int), cmpfunc);

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

	int *indices = (int *) malloc (size * sizeof(int));

	for (int i = 0; i < nchunk; ++i)
	{
		indices[i] = arr[j];
		
		printf("min_edge[%i]: %i\n", j, arr[j]);

		if (i == nchunk - 1 && j < nv)
		{
			int chunk2 = chunk / 10;
			int z = chunk2;

			for (int k = 1; k < 10; ++k)
			{
				indices[i+k] = arr[j+z];
				printf("min_edge[%i]: %i\n", j+z, arr[j+z]);
				z += chunk2;

			}
		}
		

		j+=chunk;
 	}

 	int new_size = nchunk*2;
 	 	
 	int r = eliminate_repeated_elements_from_array (indices, size, new_size, min_edge);

 	return r;

}


