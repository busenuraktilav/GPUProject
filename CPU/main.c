#include <stdio.h>
#include <stdlib.h>


#include "bellman_ford.h"
#include "dijkstra.h"
#include "hybrid.h"
#include "cpu_utils.h"



int main(int argc, char const *argv[])
{

  	bool signal_originalDistance = atoi(argv[1]);
	bool signal_kernelMinEdge = atoi(argv[2]);
	bool signal_appr_attr = atoi(argv[3]);
	bool signal_reduce_execution = atoi(argv[4]);
	int signal_partial_graph_process = atoi(argv[5]);
	bool signal_atomicMinBlock = atoi(argv[6]);
	bool signal_atomicMaxBlock = atoi(argv[7]);
	bool signal_atomicAddBlock = atoi(argv[8]);
	bool signal_atomicExchBlock = atoi(argv[9]);

	const char* file = argv[10];
	
	float min_edges_to_process = atoi(argv[11]);
	float iter_num = atoi(argv[12]);
	float percentage = atof(argv[13]);
	bool write = atoi(argv[14]);
	int algorithm_option = atoi(argv[15]);

	switch (algorithm_option) 
	{
		case 1:

			main_bellman(signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, 
		                 signal_reduce_execution, signal_partial_graph_process, 
		                 signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock, 
		                 signal_atomicExchBlock, file, min_edges_to_process, iter_num, 
		                 percentage, write);
			break;

		case 2:

			main_dijkstra(signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, 
		                  signal_reduce_execution, signal_partial_graph_process, 
		                  signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock,
		                  signal_atomicExchBlock, file, min_edges_to_process, iter_num, 
		                  percentage, write);
			break;

		case 3:

			main_hybrid(signal_originalDistance, signal_kernelMinEdge, signal_appr_attr, 
		                  signal_reduce_execution, signal_partial_graph_process, 
		                  signal_atomicMinBlock, signal_atomicMaxBlock, signal_atomicAddBlock,
		                  signal_atomicExchBlock, file, min_edges_to_process, iter_num, 
		                  percentage, write);
			break;

		default:

			printf("Algorithm_option is not passed to main function!\n");
			break;

	}


	return 0;
}
