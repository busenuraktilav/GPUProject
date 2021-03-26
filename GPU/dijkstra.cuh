void sdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source, int *count, double *time);
void apprsdj(const int *row_ptr, const int *col_ind, const int *weights, int **distance, int **previous, const int nv, const int ne, int source, float **appr_vals, int max_distance, float *time);
