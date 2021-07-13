main: ../../../include/graphio/graphio.c ../../../include/utils/cpu_utils.c dijkstra_appr.cu main.c
	nvcc -o run -O3 -lm -lcurand -lcuda -gencode=arch=compute_60,code=sm_60 ../../../include/graphio/graphio.c ../../../include/utils/cpu_utils.c dijkstra_appr.cu main.c
