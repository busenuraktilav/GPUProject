void shybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	     const int nv, const int ne, int source, int *signals, float **signal_variables,
	     int neg_edge_count, double *time);
void apprshybrid(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	         const int nv, const int ne, int source, int *signals, float **signal_variables, 
	         int max_distance, int neg_edge_count, float *time);