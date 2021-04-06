main: ../graphio/graphio.c heap.c dijkstra.c bellman_ford.c main.c main_bellman.c main_dijkstra.c main_hybrid.c
	nvcc -o run -lcuda -O3 -lcurand ../graphio/graphio.c main.c heap.c dijkstra.c bellman_ford.c cpu_utils.c main_bellman.c main_dijkstra.c main_hybrid.c ../GPU/dijkstra.cu ../GPU/bellman_ford.cu ../GPU/hybrid.cu ../GPU/bellman_ford_appr.cu ../GPU/dijkstra_appr.cu ../GPU/hybrid_appr.cu
