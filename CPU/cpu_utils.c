#include "cpu_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <stdbool.h>


float relative_error(int **arr1, int **arr2, int size)
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
			
			if(!isnan(err/actual_val)){
				error += err/actual_val;
			}
		}
	}

	return (error/size)*100;
}




void init_zero(int **arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		(*arr)[i] = 0;
	}
}


//Technique 4: round weights to the nearest power of 2
void approximate_attributes(int **weights, int ne, int *max_weight, int *min_weight)
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


int sync_bfs(int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source)
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


float jaccard_similarity(int *row_ptr, int *col_ind, int v1, int v2, int *conj, int *diff)
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

void merge_nodes(int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int v1, int v2, int conj, int diff)
{
	int edge_num_v1 = row_ptr[v1+1] - row_ptr[v1];
	int edge_num_v2 = row_ptr[v2+1] - row_ptr[v2];

	int merged_edges[conj+diff];
	int merged_weights[conj+diff];
	/*
	for (int i = 0; i < edge_num_v1; i++)
	{
		for (int j = 0; j < edge_num_v2; j++)
		{

		}
	}
	*/
}


//Technique 3: Approximate graph representation
void appr_graph_rep(int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int similarity_rate)
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
				merge_nodes(row_ptr, col_ind, weights, nv, ne, v1, v2, conj, diff);
			}
		}
	}
}

int write_distance(const char *filename, int *distance, int *iter_num, int nv)
{
	FILE *fp = fopen(filename, "w");

	if(fp == NULL)
		return -1;


	fprintf(fp, "%d\n", *iter_num);

	for (int i = 0; i < nv; i++)
	{
		fprintf(fp, "%d\n", distance[i]);
	}

	fclose(fp);

	return 1;
}

int read_distance(const char *filename, int **distance, int *iter_num, int nv)
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

	for (i = 0; i < nv; ++i)
	{
		fgets(line, 1025, fp);
		sscanf(line, "%d", &d);

		(*distance)[i] = d;
	}

	return 1;
}