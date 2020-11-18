#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "hybrid.h"
//#include "bellman_ford.h"
//#include "dijkstra.h"
#include "heap.h"


void hybrid(int **row_ptr, int **col_ind, int **row_ind, int **weights, int **distance, int **previous, int nv, int ne, int start)
{
	*distance = (int *)malloc(nv * sizeof(int));
	*previous = (int *)malloc(nv * sizeof(int));

	Heap *h = create_heap(nv);

	for(int count, v = 0; v < nv; v++)
	{
		(*distance)[v] = INT_MAX;
		(*previous)[v] = -1;
		
		insert_to_heap(h, v);
	}


	h->arr[start] = create_node(start, (*distance)[start]);
	h->pos[start] = start;
	(*distance)[start] = 0;
	decrease_key(h, start, (*distance)[start]);

	int i = 0;

	do
	{
		i++;

		while(!heap_is_empty(h))
		{
			Node* minNode = pop_min(h);
			int u = minNode->vertex_num;
			int e = (*row_ptr)[u+1] - (*row_ptr)[u];

			for (int i = 0; i < e; i++)
			{
				int tempDistance = (*distance)[u] + (*weights)[(*row_ptr)[u]+i];

				h->arr[(*col_ind)[(*row_ptr)[u]+i]]->distance = tempDistance;
				decrease_key(h, (*col_ind)[(*row_ptr)[u]+i], tempDistance);

				//RELAX 
				if(tempDistance < (*distance)[(*col_ind)[(*row_ptr)[u]+i]])
				{
					(*distance)[(*col_ind)[(*row_ptr)[u]+i]] = tempDistance;
					(*previous)[(*col_ind)[(*row_ptr)[u]+i]] = u;
				}
			}
		}

	} while (!heap_is_empty(h) || i == nv - 1);

	