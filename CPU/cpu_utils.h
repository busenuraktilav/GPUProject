#include <stdbool.h>

float relative_error(int **arr1, int **arr2, int size);
void init_zero(int **arr, int size);
void approximate_attributes(int **weights, int ne, int *max_weight, int *min_weight);
int sync_bfs(int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source);
float jaccard_similarity(int *row_ptr, int *col_ind, int v1, int v2, int *conj, int *diff);
void appr_graph_rep(int **row_ptr, int **col_ind, int **weights, int *nv, int *ne, int similarity_rate);
int write_distance(const char *filename, int *distance, int *iter_num, int *max_degree, int nv);
int read_distance(const char *filename, int **distance, int *iter_num, int *max_degree, int nv);
int write_time_results(const char *filename, float time);
float find_max_time (float *time, int repeat);
float find_min_time (float *time, int repeat, float max);
int read_time (const char* filename, float **time, int repeat);
float find_avg_time (const char *filename, int repeat);
int write_performance_results(const char *perf_file, const char *time_file, int nv, int ne, int iter_num, int max_degree, 
							  int min_edge, float percentage, bool signal_originalDistance, bool signal_kernelMinEdge, 
							  bool signal_appr_attr, bool signal_reduce_execution, bool signal_partial_graph_process,
							  float error);
