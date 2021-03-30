#include <stdbool.h>
void bellman_ford(int **row_ptr, int **col_ind, int **row_ind, int **weights, int **distance, int **previous, 
	              int nv, int ne, int start);
void main_bellman(bool signal_originalDistance, bool signal_kernelMinEdge, bool signal_appr_attr, 
	             bool signal_reduce_execution, int signal_partial_graph_process, const char *file,
	             float min_edges_to_process, float iter_num, float percentage, bool write);
