float apprshybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	         const int nv, const int ne, int source, int iter_num, int min_edges, int max_edges, 
	         float percentage, int neg_edge_count);