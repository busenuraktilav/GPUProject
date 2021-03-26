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

	for (int round = 1; round < nv; round++)
	{
		has_change = false;
		//printf("traverse %i\n", round);

		for (int v2 = 0; v2 < nv; v2++)
		{
			//int edgeStart = (*row_ptr)[v2];
			//int edgeEnd;

			//if ((v2 + 1) < nv) edgeEnd = (*row_ptr)[v2+1];
			//else               edgeEnd = ne;

			//for (int e = edgeStart; e < edgeEnd; e++)
			for (int e = (*row_ptr)[v2]; e < (*row_ptr)[v2+1]; e++)
			{
				//printf("distance[%i]: %i\n", v2, (*distance)[v2]);
				//int tempDistance = (*distance)[v2] + (*weights)[e];
				//int neigh = (*col_ind)[e];

				//printf("tempDistance: %i\n", tempDistance);
				//printf("distance[%i]: %i, weights[%i]: %i, tempDistance: %i\n", v2, (*distance)[v2], e, (*weights)[e], tempDistance);

				
				int w = (*weights)[e];
				int du = (*distance)[v2];
				int dv = (*distance)[(*col_ind)[e]];
				//printf("v2: %i, du: %i, dv: %i, w: %i add: %i\n", v2, du, dv, w, du+w);

				//if (tempDistance < (*distance)[neigh] && (*distance)[v2] != INT_MAX)
				if (du + w < dv && du != INT_MAX)
				{
					(*distance)[(*col_ind)[e]] = du + w;

					//(*distance)[neigh] = tempDistance;
					//(*previous)[neigh] = v2;
					has_change = true;
					//printf("inside RELAX\n");
					//printf("%i\n", tempDistance);
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

	/*
	
	for(int i = 0; i < 20; i++)
	{
		printf("(*distance)[%i]: %i\n", i, (*distance)[i]);
	}
	*/
	
}
