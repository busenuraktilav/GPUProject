main: ../graphio/graphio.c heap.c dijkstra.c bellman_ford.c main.c main_bellman.c main_dijkstra.c
	nvcc -o run -lcuda -O3 -lcurand ../graphio/graphio.c main.c heap.c dijkstra.c bellman_ford.c hybrid.c cpu_utils.c main_bellman.c main_dijkstra.c ../GPU/dijkstra.cu ../GPU/bellman_ford.cu ../GPU/hybrid.cu ../GPU/bellman_ford_appr.cu ../GPU/dijkstra_appr.cu
