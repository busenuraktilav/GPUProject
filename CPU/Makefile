main: graphio.c heap.c dijkstra.c bellman_ford.c main.c
	nvcc -o run -lcuda -O3 ../graphio/graphio.c heap.c dijkstra.c bellman_ford.c main.c hybrid.c cpu_utils.c ../GPU/dijkstra.cu ../GPU/bellman_ford.cu ../GPU/hybrid.cu ../GPU/bellman_ford_appr.cu
