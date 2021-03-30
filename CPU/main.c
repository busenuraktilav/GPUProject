#include <stdio.h>
#include <stdlib.h>


#include "bellman_ford.h"
#include "dijkstra.h"
#include "cpu_utils.h"



int main(int argc, char const *argv[])
{

  	bool signal_originalDistance = atoi(argv[1]);
	bool signal_kernelMinEdge = atoi(argv[2]);
	bool signal_appr_attr = atoi(argv[3]);
	bool signal_reduce_execution = atoi(argv[4]);
	int signal_partial_graph_process = atoi(argv[5]);
	const char* file = argv[6];
	float min_edges_to_process = atoi(argv[7]);
	float iter_num = atoi(argv[8]);
	float percentage = atoi(argv[9]);
	bool write = atoi(argv[10]);
	int algorithm_option = atoi(argv[11]);



	switch (algorithm_option) 
	{
		case 1:

			main_bellman(signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, 
		                 signal_reduce_execution, signal_partial_graph_process, file,
		                 min_edges_to_process, iter_num, percentage, write);
			break;

		case 2:

			main_dijkstra(signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, 
		                 signal_reduce_execution, signal_partial_graph_process, file,
		                 min_edges_to_process, iter_num, percentage, write);
			break;

		default:

			printf("Algorithm_option is not passed to main function!\n");
			break;

	}


	return 0;
}
