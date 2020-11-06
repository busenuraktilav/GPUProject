main: ./graphio/graphio.c ./CPU/heap.c ./CPU/dijkstra.c ./CPU/bellman_ford.c ./CPU/main.c
	gcc ./graphio/graphio.c ./CPU/heap.c ./CPU/dijkstra.c ./CPU/bellman_ford.c ./CPU/main.c -o run
