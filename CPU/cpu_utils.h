float relative_error(int **arr1, int **arr2, int size);
void init_zero(int **arr, int size);
void approximate_attributes(int **weights, int ne, int *max_weight, int *min_weight);
int sync_bfs(int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source);
float jaccard_similarity(int *row_ptr, int *col_ind, int v1, int v2, int *conj, int *diff);
void merge_nodes(int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int v1, int v2, int conj, int diff);
void appr_graph_rep(int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int similarity_rate);
int write_distance(const char *filename, int *distance, int *iter_num, int nv);
int read_distance(const char *filename, int **distance, int *iter_num, int nv);
