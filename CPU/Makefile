main: graphio.c heap.c dijkstra.c bellman_ford.c main.c
	nvcc -o run -lcuda ../graphio/graphio.c heap.c dijkstra.c bellman_ford.c mergedmain.c ../GPU/dijkstra.cu ../GPU/bellman_ford.cu
