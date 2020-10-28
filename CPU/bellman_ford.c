void bellman_ford(int **row_ptr, int **col_ind, int **row_ind, int **weights, int nv, int ne, int start)
{
	
	int *distance = (int *)malloc(nv * sizeof(int));
	memset(distance, INF, sizeof(int) * nv);
	int *previous = (int *)malloc(nv * sizeof(int));
	memset(previous, -1, sizeof(int) * nv);

	(distance)[start] = 0;


	for (int v1 = 0; v1 < nv; v1++)
	{
		for(int v2 = 0; v2 < ne; v2++)
		{
			//RELAX procedure
			int tempDistance = distance[(*row_ind)[v2]] + (*weights)[v2];

			if(tempDistance < distance[(*col_ind)[v2]])
			{
				distance[(*col_ind)[v2]] = tempDistance;
				previous[(*col_ind)[v2]] = (*row_ind)[v2];
			}
			//printf("%i-%i -> %i\n", (*row_ind)[v2], (*col_ind)[v2], distance[(*col_ind)[v2]]);
		}
	}

	//check for negative cycles
	for(int v2 = 0; v2 < ne; v2++)
	{
		if(distance[(*row_ind)[v2]] + (*weights)[v2] < distance[(*col_ind)[v2]])
			printf("Error: negative cycle exists\n");
	}

	/*

	for(int i = 0; i < nv; i++){
		printf("%i\n", distance[i]);
	}

	for(int i = 0; i < nv; i++){
		printf("%i\n", previous[i]);
	}

	*/
}
