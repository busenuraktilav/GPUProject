#include <stdbool.h>
//void hybrid(int **row_ptr, int **col_ind, int **row_ind, int **weights, int **distance, 
//	        int **previous, int nv, int ne, int start, int *count, int neg_edge_count);
int main_hybrid(bool signal_originalDistance, bool signal_kernelMinEdge, bool signal_appr_attr, 
	            bool signal_reduce_execution, int signal_partial_graph_process, const char *file,
	            float min_edges_to_process, float iter_num, float percentage, bool write);
