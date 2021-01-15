main: graphio.c heap.c dijkstra.c bellman_ford.c main.c
	nvcc -o run -lcuda -O3 ../graphio/backup2_graphio.c heap.c dijkstra.c bellman_ford.c mergedmain.c hybrid.c ../GPU/dijkstra.cu ../GPU/bellman_ford.cu ../GPU/hybrid.cu
