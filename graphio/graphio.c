#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

#include "graphio.h"

struct edge
{
	int i;
	int j;
	int w;
};

typedef struct edge Edge;


int edge_cmp(const void *e1, const void *e2)
{
	Edge *edge1 = (Edge *)e1;
	Edge *edge2 = (Edge *)e2;

	if(edge1->i == edge2->i)

		return (int)(edge1->j - edge2->j);

	else
		return (int)(edge1->i - edge2->i);
}


int read_graph(const char *file, int **row_ptr, int **col_ind, int **row_ind, int **weights, int *nv, int *ne, int *neg_edge_count, int *max_weight, int *min_weight){

	printf("Reading input file %s\n", file);

	FILE *fp = fopen(file, "r");
	char line[1025];
	int m = 0, n = 0, nnz = 0;
	*max_weight = INT_MIN;
	*min_weight = INT_MAX;

	
	if(fp == NULL)
	{
		printf("file could not be found\n");
		return -1;
	}

	do
	{
		if(fgets(line, 1025, fp) == NULL)
		{
			printf("file does not contain matrix\n");
			return -1;
		}
	
	} while(line[0] == '%');
	
	sscanf(line, "%d %d %d", &m, &n, &nnz);

	printf("---------%i, %i %i\n", m, n, nnz);

	if(m != n)
	{
		printf("Not a square matrix\n");
		return -1;
	}


	long long int size_needed = 2 * nnz * sizeof(Edge);
	Edge *E = (Edge *)malloc(size_needed);

	int cnt = 0, ecnt, i, j, w;
	int self_loop_flag = 1;

	
	for (ecnt = 0; ecnt < nnz; ++ecnt)
	{
		fgets(line, 1025, fp);

		sscanf(line, "%d %d %d", &i, &j, &w);

		if(i != j) //no self loops
		{
			E[cnt].i = i;
			E[cnt].j = j;
			E[cnt].w = w;
			cnt++;

			if (w < 0)
				(*neg_edge_count)++;

			if (w > *max_weight)
				*max_weight = w;

			if (w < *min_weight)
				*min_weight = w;
 		}


 		else if(self_loop_flag && i == j)
 		{
 			self_loop_flag = 0;
 			printf("Warning: Graph contains self loops\n");
 		}

 		
	}


	qsort(E, cnt, sizeof(Edge), edge_cmp);

	*row_ind = (int *)calloc((m+2), sizeof(int));

	*row_ptr = (int *)calloc((m+2), sizeof(int));
	(*row_ptr)[E[0].i + 1]++;

	*col_ind = (int *)malloc(cnt*sizeof(int));
	(*col_ind)[0] = E[0].j;

	*weights = (int *)malloc(cnt*sizeof(int));
	(*weights)[0] = E[0].w;

	int k = 0, l = 1, h = 1; 

	for (ecnt = 1; ecnt < cnt; ecnt++)
	{
		i = E[ecnt].i;
        if (i != E[ecnt - 1].i || E[ecnt].j != E[ecnt - 1].j) 
        { 
            (*row_ptr)[i + 1]++;
            k = i; // the first edge entry 
            (*col_ind)[l++] = E[ecnt].j;
            (*weights)[h++] = E[ecnt].w;
        }
    }

    
    
	for (i = 2; i <= m+1; ++i) // cumulative sum
	{ 
        (*row_ptr)[i] += (*row_ptr)[i - 1];
    }

    

    *nv = m;
    *ne = cnt;

   
    printf("graph reading is DONE!\n");

    return 1;   
	
}


