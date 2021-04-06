void shybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, int **previous, 
	         const int nv, const int ne, int source, int *count, int neg_edge_count);
void apprshybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	             int **previous, const int nv, const int ne, int source, 
	             int neg_edge_count, float **appr_vals, int max_distance, float *time);
	             