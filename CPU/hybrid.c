#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "hybrid.h"
#include "heap.h"


/*A hybrid of the Bellman–Ford and Dijkstra algorithms is suggested, improving the
running time bound of Bellman–Ford for graphs with a sparse distribution of negative cost
edges. The algorithm iterates Dijkstra several times without re-initializing the tentative
value d ( v ) at vertices. At most k + 2 iterations solve the problem, if for any vertex reachable
from the source, there exists a shortest path to it with at most k negative cost edges.*/
void hybrid(int **row_ptr, int **col_ind, int **weights, int **distance, int nv, 
	        int ne, int start)
{
	*distance = (int *)malloc(nv * sizeof(int));
	
	for(int count, v = 0; v < nv; v++)
	{
		(*distance)[v] = INT_MAX;
	}

	int k = 1;//at most k negative edges
	//for(int iteration = 0; iteration < 10; iteration++)
	for(int iteration = 0; iteration < k + 2; iteration++)
	{
		Heap *h = create_heap(nv);

		for(int count, v = 0; v < nv; v++)
		{
			insert_to_heap(h, v);
		}

		//printf("Dijkstra run %d\n", iteration);
		h->arr[start] = create_node(start, (*distance)[start]);
		h->pos[start] = start;
		(*distance)[start] = 0;
		decrease_key(h, start, (*distance)[start]);

		while(!heap_is_empty(h))
		{
			Node* minNode = pop_min(h);
			int u = minNode->vertex_num;
			int e = (*row_ptr)[u+1] - (*row_ptr)[u];

			for (int i = 0; i < e; i++)
			{
				int tempDistance = (*distance)[u] + (*weights)[(*row_ptr)[u]+i];
				int dest = (*col_ind)[(*row_ptr)[u]+i];
				if((*distance)[u] != INT_MAX)
				{
					h->arr[dest]->distance = tempDistance;
					decrease_key(h, dest, tempDistance);
				}

				if(tempDistance < (*distance)[dest] && (*distance)[u] != INT_MAX)
				{
					(*distance)[dest] = tempDistance;
					//printf("Temp distance update at iteration %d\n",iteration);
				}
			}
		}

	} 
}