#include <stdbool.h>
void bellman_ford(int **row_ptr, int **col_ind, int **weights, int **distance, 
	              int nv, int ne, int start);