#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "../graphio/graphio.h"


int main(int argc, char const *argv[])
{
	const char *file = argv[1];
	char f[2048];

	strcpy(f, file);

	char *graph_name = strtok(f, ".");

	char bfileneg[2048];
    FILE *bp, *fp, *bpneg;
    sprintf(bfileneg, "%s_neg.bin", graph_name);

    if (fopen(bfileneg, "rb") == NULL)
    {
    	int *row_ptr, *col_ind, *weights, *distance, *approximate_distance, max_weight, 
	    min_weight, nv, ne, iter_num = 0, max_degree = 0, neg_edge_count = 0;

		int read = read_graph(file, &row_ptr, &col_ind, &weights, &nv, &ne, &neg_edge_count, &max_weight, &min_weight);

		if(read != 1)
		{
			printf("Something went terribly wrong while reading the graph\n");
			return -1;
		}

	    int w, r;
	    neg_edge_count = ne/5000000;
	    printf("neg_edge_count: %i\n", neg_edge_count);

	    srand(time(NULL));

	    for (int i = 0; i < 1; ++i)
	    {

	    	//r = rand()%(ne-1);
	    	w = weights[i+5];
	    	weights[i+5] = -w;

	    	printf("weights[%i]: %i, weights[%i]: %i\n", i+5, weights[i+5], i+6, weights[i+6]);
	    }

	    bpneg = fopen(bfileneg, "wb");

	    if (bpneg != NULL)
	    {
	        write_graph_binary(bpneg, row_ptr, col_ind, weights, nv, ne, neg_edge_count);
	        fclose(bpneg);
	    }
	    printf("Negative graph created!\n");
    }

    else
    	printf("Negative graph already exists!\n");



	
	return 0;
}