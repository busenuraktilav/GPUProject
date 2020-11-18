#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "bellman_ford.h"


void bellman_ford(int **row_ptr, int **col_ind, int **row_ind, int **weights, int **distance, int **previous, int nv, int ne, int start)
{
	*distance = (int *)malloc(nv * sizeof(int));
	*previous = (int *)malloc(nv * sizeof(int));

	for(int i = 0; i < nv; i++)
	{
		(*distance)[i] = INT_MAX;
		(*previous)[i] = -1;
	}


	(*distance)[start] = 0;

	for(int v1 = 0; v1 < nv; v1++)
	{
		for(int v2 = 0; v2 < ne; v2++)
		{
			//RELAX procedure
			int tempDistance = (*distance)[(*row_ind)[v2]] + (*weights)[v2];

			if(tempDistance < (*distance)[(*col_ind)[v2]] && (*distance)[(*row_ind)[v2]] != INT_MAX)
			{
				(*distance)[(*col_ind)[v2]] = tempDistance;
				(*previous)[(*col_ind)[v2]] = (*row_ind)[v2];
			}

			//printf("%i-%i -> %i\n", (*row_ind)[v2], (*col_ind)[v2], (*distance)[(*col_ind)[v2]]);
		}
	}

	//check for negative cycles
	for(int v2 = 0; v2 < ne; v2++)
	{
		if((*distance)[(*row_ind)[v2]] + (*weights)[v2] < (*distance)[(*col_ind)[v2]] && (*distance)[(*row_ind)[v2]] != INT_MAX)
		{
			printf("belman ford Error: negative cycle exists\n");
			break;
		}
	}

}
