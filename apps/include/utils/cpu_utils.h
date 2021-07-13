#include <stdbool.h>

float relative_error(int **arr1, int **arr2, int size);
void init_zero(int **arr, int size);
void approximate_attributes(int **weights, int ne, int *max_weight, int *min_weight);
int sync_bfs(int *row_ptr, int *col_ind, int nv, int ne, int *level_arr, int source);
int write_time_results(const char *filename, float time);
float find_max_time (float *time, int repeat);
float find_min_time (float *time, int repeat, float max);
int count_lines (const char *filename);
int read_time (const char* filename, float **time, int *repeat);
float find_avg_time (const char *filename);
int write_performance_results(const char *perf_file, const char *time_file, const char *error_file, int nv, int ne, 
	                          int max_degree, int *signals, int min_edges, int max_edges,
	                          int iter_num, float percentage);
int cmpfuncmin (const void * a, const void * b);
int cmpfuncmax (const void * a, const void * b);
int eliminate_repeated_elements_from_array (const int *arr, int size, int new_size, int min_edge);
int min_edge_to_process (const int *row_ptr, int nv, int min_edges);
int max_edge_to_process (const int *row_ptr, int nv, int max_edge);
int select_start_vertex(int *row_ptr, int nv, int *max_degree);
