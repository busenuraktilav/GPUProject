#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

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

	bool has_change;

	for (int v1 = 0; v1 < nv-1; v1++)
	{
		has_change = false;
		//printf("traverse %i\n", v1);

		for (int v2 = 0; v2 < nv; v2++)
		{
			int edgeStart = (*row_ptr)[v2];
			int edgeEnd;

			if ((v2 + 1) < nv) edgeEnd = (*row_ptr)[v2+1];
			else               edgeEnd = ne;

			for (int e = edgeStart; e < edgeEnd; e++)
			{
				//printf("distance[%i]: %i\n", v2, (*distance)[v2]);
				int tempDistance = (*distance)[v2] + (*weights)[e];
				int neigh = (*col_ind)[e];

				//printf("tempDistance: %i\n", tempDistance);

				if (tempDistance < (*distance)[neigh] && (*distance)[v2] != INT_MAX)
				{
					(*distance)[neigh] = tempDistance;
					(*previous)[neigh] = v2;
					has_change = true;
					//printf("inside RELAX\n");
				}
			}
		}

		if(!has_change)
			break;
	}

	//check for negative cycles

	int neg_cycle = false;

	for (int v2 = 0; v2 < nv; v2++)
	{
		int edgeStart = (*row_ptr)[v2];
		int edgeEnd;

		if((v2+1) < nv) edgeEnd = (*row_ptr)[v2+1];
		else            edgeEnd = ne;

		for (int e = edgeStart; e < edgeEnd; e++)
		{
			int tempDistance = (*distance)[v2] + (*weights)[e];
			int neigh = (*col_ind)[e];

			if (tempDistance < (*distance)[neigh] && (*distance)[v2] != INT_MAX)
			{
				printf("belman ford Error: negative cycle exists\n");
				neg_cycle = true;
				break;

			}

		}

		if(neg_cycle)
			break;
	}

}
