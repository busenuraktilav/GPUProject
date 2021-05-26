#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>


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


int process_graph(FILE *fp, int **row_ptr, int **col_ind, int **weights, int *nv, 
	           int *ne, int *neg_edge_count, int *max_weight, int *min_weight)
{
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

	printf("##### %i, %i %i #####\n", m, n, nnz);

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

	//*row_ind = (int *)calloc((m+2), sizeof(int));

	*row_ptr = (int *)calloc((m+2), sizeof(int));
	(*row_ptr)[E[0].i + 1]++;

	*col_ind = (int *)malloc(cnt*sizeof(int));
	(*col_ind)[0] = E[0].j;

	*weights = (int *)malloc(cnt*sizeof(int));
	(*weights)[0] = E[0].w;

	int l = 1, h = 1;

	for (ecnt = 1; ecnt < cnt; ecnt++)
	{
		i = E[ecnt].i;

        if (i != E[ecnt - 1].i || E[ecnt].j != E[ecnt - 1].j) 
        { 
            (*row_ptr)[i + 1]++;
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


int read_graph_binary(FILE *bp, int **row_ptr, int **col_ind, int **weights, int *nv)
{
	fread(nv, sizeof(int), 1, bp);

    *row_ptr = (int *)malloc(sizeof(int) * (*nv + 1));
    fread(*row_ptr, sizeof(int), (size_t)(*nv + 1), bp);

    *col_ind = (int *)malloc(sizeof(int) * (*row_ptr)[*nv]);
    fread(*col_ind, sizeof(int), (size_t)(*row_ptr)[*nv], bp);

    *weights = (int *)malloc(sizeof(int) * (*row_ptr)[*nv]);
    fread(*weights, sizeof(int), (size_t)(*row_ptr)[*nv], bp);

    return 1;

}

int write_graph_binary(FILE *bp, int *row_ptr, int *col_ind, int *weights, int nv)
{
    fwrite(&nv, sizeof(int), (size_t)1, bp);
    fwrite(row_ptr, sizeof(int), (size_t)(nv + 1), bp);
    fwrite(col_ind, sizeof(int), (size_t)(row_ptr[nv]), bp);
    fwrite(weights, sizeof(int), (size_t)(row_ptr[nv]), bp);

    return 1;

}

int read_graph(const char *file, int **row_ptr, int **col_ind, int **weights, 
	              int *nv, int *ne, int *neg_edge_count, int *max_weight, 
	              int *min_weight)
{
    char bfile[2048];
    FILE *bp, *fp;
    sprintf(bfile, "%s.bin", file);
    bp = fopen(bfile, "rb");

    if (bp != NULL) // read binary file 
    {
        printf("Reading binary file\n");
        read_graph_binary(bp, row_ptr, col_ind, weights, nv);
        
        fclose(bp);
    }
    else // read from text
    {
        fp = fopen(file, "r");

        if (fp == NULL)
        {
            printf("%s, file does not exist\n", file);
            return -1;
        }

        else
        {
            process_graph(fp, row_ptr, col_ind, weights, nv, ne, neg_edge_count, max_weight, min_weight);

            bp = fopen(bfile, "wb");

            if (bp != NULL)
            {
                write_graph_binary(bp, *row_ptr, *col_ind, *weights, *nv);
                fclose(bp);
            }

            fclose(fp);

        }
    }

    *ne = (*row_ptr)[*nv];

    return 1;

}


int write_distance (const char *filename, int *distance, int iter_num, int nv)
{
	FILE *fp = fopen(filename, "w");

	if(fp == NULL)
		return -1;


	fprintf(fp, "%d\n", iter_num);
	//fprintf(fp, "%d\n", *max_degree);

	for (int i = 0; i < nv; i++)
	{
		fprintf(fp, "%d\n", distance[i]);
	}

	fclose(fp);

	return 1;
}

int read_distance (const char *filename, int **distance, int *iter_num, int nv)
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
	//fgets(line, 1025, fp);
	//sscanf(line, "%d", max_degree);

	for (i = 0; i < nv; ++i)
	{
		fgets(line, 1025, fp);
		sscanf(line, "%d", &d);

		(*distance)[i] = d;
	}

	return 1;
}