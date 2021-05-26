void sdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	     const int nv, const int ne, int source, int *signals, float **signal_variables,
	     double *time);
void apprsdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, 
	         const int nv, const int ne, int source, int *signals, float **signal_variables, 
	         int max_distance, float *time);