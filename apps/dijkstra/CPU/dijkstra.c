#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "dijkstra.h"
#include "../../include/utils/heap.h"

void dijkstra(int **row_ptr, int **col_ind, int **weights, int **distance, int nv, 
	          int ne, int start)
{
	*distance = (int *)malloc(nv * sizeof(int));

	Heap *h = create_heap(nv+1);

	for(int count, v = 0; v < nv+1; v++)
	{
		(*distance)[v] = INT_MAX;
		insert_to_heap(h, v);
	}


	h->arr[start] = create_node(start, (*distance)[start]);
	h->pos[start] = start;
	(*distance)[start] = 0;
	decrease_key(h, start, (*distance)[start]);


	while(!heap_is_empty(h))
	{
		Node* minNode = pop_min(h);
		int u = minNode->vertex_num;
		int e = (*row_ptr)[u+1] - (*row_ptr)[u];

		//printf("size of heap: %i, min_node_vertex_num: %i, minNode->distance: %i, e: %i (*row_ptr)[u+1]: %i, (*row_ptr)[u]: %i\n", h->count, u, minNode->distance, e, (*row_ptr)[u+1], (*row_ptr)[u]);

		if (minNode->distance == INT_MAX)
			break;

		for (int i = 0; i < e; i++)
		{
			//printf("inside cpu dijkstra: %i, (*distance)[%i]: %i\n", i, u, (*distance)[u]);
			int tempDistance = (*distance)[u] + (*weights)[(*row_ptr)[u]+i];

			if((*distance)[u] != INT_MAX)
			{
				h->arr[(*col_ind)[(*row_ptr)[u]+i]]->distance = tempDistance;
				decrease_key(h, (*col_ind)[(*row_ptr)[u]+i], tempDistance);
			}

			if(tempDistance < (*distance)[(*col_ind)[(*row_ptr)[u]+i]] && (*distance)[u] != INT_MAX)
			{
				(*distance)[(*col_ind)[(*row_ptr)[u]+i]] = tempDistance;
			}
		}
	}
}