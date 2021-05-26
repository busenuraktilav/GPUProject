#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#include "bellman_ford.h"


void bellman_ford(int **row_ptr, int **col_ind, int **weights, int **distance, int nv, 
	              int ne, int start)
{
	*distance = (int *)malloc(nv * sizeof(int));

	for(int i = 0; i < nv; i++)
	{
		(*distance)[i] = INT_MAX;
	}


	(*distance)[start] = 0;

	bool has_change;

	for (int round = 1; round < nv; round++)
	{
		has_change = false;

		for (int v2 = 0; v2 < nv; v2++)
		{
			
			for (int e = (*row_ptr)[v2]; e < (*row_ptr)[v2+1]; e++)
			{
				
				int w = (*weights)[e];
				int du = (*distance)[v2];
				int dv = (*distance)[(*col_ind)[e]];
				
				if (du + w < dv && du != INT_MAX)
				{
					(*distance)[(*col_ind)[e]] = du + w;
					has_change = true;
				}
			}
		}

		if(!has_change){
			printf("round for cpu: %i\n", round);
			break;
		}
	}

	//check for negative cycles

	int neg_cycle = false;

	for (int v2 = 0; v2 < nv; v2++)
	{
		for (int e = (*row_ptr)[v2]; e < (*row_ptr)[v2+1]; e++)
		{
			int tempDistance = (*distance)[v2] + (*weights)[e];
			int neigh = (*col_ind)[e];

			if (tempDistance < (*distance)[neigh] && (*distance)[v2] != INT_MAX)
			{
				printf("CPU belman ford Error: negative cycle exists\n");
				neg_cycle = true;
				break;

			}

		}

		if(neg_cycle)
			break;
	}
	
}
