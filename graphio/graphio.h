#ifndef GRAPHIO_H
#define GRAPHIO_H

int read_graph(const char *file, int **row_ptr, int **row_ind, int **col_ind, int **weights, int *nv, int *ne);

#endif GRAPHIO_H