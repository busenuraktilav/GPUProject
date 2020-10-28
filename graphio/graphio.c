#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>

#include "graphio.h"

//#define INFINITY 99999
//#define INF 9999
#define INF 0x3f3f3f3f

int read_graph(const char *file, int **row_ptr, int **col_ind, int **row_ind, int **weights, int *nv, int *ne){

	printf("Reading input file %s\n", file);

	FILE *fp = fopen(file, "r");
	char line[1025];
	int m = 0, n = 0, nnz = 0;
	
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

	if(m != n)
	{
		printf("Not a square matrix\n");
		return -1;
	}


	*row_ptr = (int *)malloc(sizeof(int) * (m+1));
	memset((*row_ptr), 0, sizeof(int) * (m + 1));
	*col_ind = (int *)malloc((nnz+1) * sizeof(int));
	memset((*col_ind), 0, sizeof(int) * (nnz + 1));
	*row_ind = (int *)malloc((nnz+1) * sizeof(int));
	memset((*row_ind), 0, sizeof(int) * (nnz + 1));
	*weights = (int *)malloc((nnz+1) * sizeof(int));
	memset((*weights), 0, sizeof(int) * (nnz + 1));
	/*
	for(int i = 0; i < nnz; i++)
	{
		(*weights)[i] = INF;
	}*/

	
	//int size_needed = (m+1) * (n+1) * sizeof(int);
	int size_needed = m * n * sizeof(int);
	int *arr = (int *)malloc(size_needed);
	memset((arr), 0, size_needed);
	int i, j, w, cnt, cnt2;
	bool self_loop_flag = false;

	for(cnt = 0; cnt < nnz; cnt++)
	{
		fgets(line, 1025, fp);
		sscanf(line, "%d %d %d", &i, &j, &w);

		if(i == j)
		{
			self_loop_flag = true;
			continue;
		}

		*(arr + (i-1)*m + (j-1)) = w;
		//*(arr + i*m + j) = w;
		//printf("%i %i %i\n", i, j, w);
	}

	if(self_loop_flag)
	{
		printf("Warning: Graph contains self loops\n");
	}

	printf("---------%i, %i %i\n", m, n, nnz);
	

	int ci = 0, cr = 1, fr = 0;
	(*row_ptr)[0] = 0;

	for(cnt = 0; cnt < m; cnt++)
	{
		for(cnt2 = 0; cnt2 < n; cnt2++)
		{
			if(*(arr + cnt*m + cnt2) != 0)
			{
				(*col_ind)[ci] = cnt2;
				//printf("%i\n", cnt2);
				(*weights)[ci] = *(arr + cnt*m + cnt2);
				//printf("%i\n", (*weights)[ci]);
			
				ci++;
				
			}
		}

		(*row_ptr)[cr] = ci; 

		for(int i = (*row_ptr)[cr-1]; i < (*row_ptr)[cr]; i++)
		{
			(*row_ind)[i] = cnt;
			//printf("%i %i %i %i %i\n", (*row_ptr)[cr-1], (*row_ptr)[cr], i, cnt, (*row_ind)[cnt]);
		}
		//printf("%i\n", ci);
		//printf("%i\n", ci + 1);
		cr++;

	}
	/*
	for(cnt = 0; cnt < nnz; cnt++)
	{
		//printf("%i %i\n", cnt, (*row_ind)[cnt]);
	}
	*/

	*nv = m;
	*ne = nnz;
	
	return 1;
}

