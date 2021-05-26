int process_graph(FILE *fp, int **row_ptr, int **col_ind, int **weights, int *nv, 
	           int *ne, int *neg_edge_count, int *max_weight, int *min_weight);
int read_graph_binary(FILE *bp, int **row_ptr, int **col_ind, int **weights, int *nv);
int write_graph_binary(FILE *bp, int *row_ptr, int *col_ind, int *weights, int nv);
int read_graph(const char *file, int **row_ptr, int **col_ind, int **weights, 
	              int *nv, int *ne, int *neg_edge_count, int *max_weight, 
	              int *min_weight);
int write_distance(const char *filename, int *distance, int iter_num, int nv);
int read_distance(const char *filename, int **distance, int *iter_num, int nv);