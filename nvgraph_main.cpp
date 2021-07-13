#include <nvgraph.h>
#include <cstdio>
#include <cstdlib>
#include <libgen.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <cuda_profiler_api.h>

#include <time.h>




void check(nvgraphStatus_t status) {

        if (status != NVGRAPH_STATUS_SUCCESS) {
                printf("ERROR : %d\n",status);
                exit(0);
        }
}

struct edge
{
        int i;
        int j;
        int w;
};

typedef struct edge Edge;


int edge_cmp(const void *e1, const void *e2)
{
        Edge *edge1 = (Edge *)e1;
        Edge *edge2 = (Edge *)e2;

        if(edge1->j == edge2->j){

                return (int)(edge1->i - edge2->i);
        }

        else{
                return (int)(edge1->j - edge2->j);
        }
}

// CSC graph construct

int read_graph(const char *file, int **row_ptr, int **col_ind, int **row_ind, int **weights, int *nv, int *ne, int *neg_edge_count){

        printf("Reading input file %s\n", file);

        FILE *fp = fopen(file, "r");
        char line[1025];
        int m = 0, n = 0, nnz = 0;

        
        if(fp == NULL)
        {
                printf("file could not be found\n");
                return -1;
        }

        do
        {
                if(fgets(line, 1025, fp) == NULL)
                {
                        printf("file does not contain matrix\n");
                        return -1;
                }
        
        } while(line[0] == '%');
        
        sscanf(line, "%d %d %d", &m, &n, &nnz);

        printf("---------%i, %i %i\n", m, n, nnz);

        if(m != n)
        {
                printf("Not a square matrix\n");
                return -1;
        }


        long long int size_needed = 2 * nnz * sizeof(Edge);
        Edge *E = (Edge *)malloc(size_needed);

        int cnt = 0, ecnt, i, j, w;
        int self_loop_flag = 1;

        
        for (ecnt = 0; ecnt < nnz; ++ecnt)
        {
                fgets(line, 1025, fp);

                sscanf(line, "%d %d %d", &i, &j, &w);

                if(i != j) //no self loops
                {
                        E[cnt].i = i;
                        E[cnt].j = j;
                        E[cnt].w = w;
                        cnt++;

                        if (w < 0)
                                (*neg_edge_count)++;
                }


                else if(self_loop_flag && i == j)
                {
                        self_loop_flag = 0;
                        printf("Warning: Graph contains self loops\n");
                }
        }


        qsort(E, cnt, sizeof(Edge), edge_cmp);

        *row_ind = (int *)calloc((m+2), sizeof(int));

        *row_ptr = (int *)calloc((m+2), sizeof(int));
        (*row_ptr)[E[0].j + 1]++;

        *col_ind = (int *)malloc(cnt*sizeof(int));
        (*col_ind)[0] = E[0].i;

        *weights = (int *)malloc(cnt*sizeof(int));
        (*weights)[0] = E[0].w;

        int k = 0, l = 1, h = 1; 

        for (ecnt = 1; ecnt < cnt; ecnt++)
        {
                j = E[ecnt].j;

                if (j != E[ecnt - 1].j || E[ecnt].i != E[ecnt - 1].i) 
                { 
                    (*row_ptr)[j + 1]++;
                    //k = j; // the first edge entry 
                    (*col_ind)[l++] = E[ecnt].i;
                    (*weights)[h++] = E[ecnt].w;
                }
        }

    
    
        for (i = 2; i <= m+1; ++i) // cumulative sum
        { 
                (*row_ptr)[i] += (*row_ptr)[i - 1];
        }

    

        *nv = m;
        *ne = cnt;

   
    printf("graph reading is DONE!\n");

    return 1;   
        
}


int main(int argc, char **argv)
{

        const char* file = "./example_graphs/graph_12_16_pos-10.txt";
        //const char* file = "./example_graphs/deneme.txt";

        int *source_offsets_h, *destination_indices_h, *row_ind, *weights_h, nv, ne, neg_edge_count = 0;

        int read = read_graph(file, &source_offsets_h, &destination_indices_h, &row_ind, &weights_h, &nv, &ne, &neg_edge_count);

        if (read != 1)
        {
                printf("ERROR. Reading failed!\n");
                return 0;
        }


        const size_t  n = nv, nnz = ne, vertex_numsets = 1, edge_numsets = 1;
        int *sssp_1_h;
        void** vertex_dim;

        // nvgraph variables    
        nvgraphStatus_t status;
        nvgraphHandle_t handle;
        nvgraphGraphDescr_t graph;
        nvgraphCSCTopology32I_t CSC_input;
        cudaDataType_t edge_dimT = CUDA_R_32F;
        cudaDataType_t* vertex_dimT;

        // Init host data    
        sssp_1_h = (int*)malloc(n*sizeof(int));
        vertex_dim  = (void**)malloc(vertex_numsets*sizeof(void*));
        vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
        CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
        vertex_dim[0]= (void*)sssp_1_h;
        vertex_dimT[0] = CUDA_R_32F;
        

        check(nvgraphCreate(&handle));
        check(nvgraphCreateGraphDescr (handle, &graph));
        CSC_input->nvertices = n; CSC_input->nedges = nnz;    
        CSC_input->destination_offsets = source_offsets_h;    
        CSC_input->source_indices = destination_indices_h;

        // Set graph connectivity and properties (tranfers)    
        check(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
        check(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
        check(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
        check(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));


        // Solve

        int source_vert = 0;
        int max_degree = 0, deg;
        for (int i = 0; i < nv; ++i)
        {
                deg = source_offsets_h[i + 1] - source_offsets_h[i];
                if (deg > max_degree)
                {
                        source_vert = i;
                        max_degree = deg;
                }
        }

        printf("Start node: %i\n", source_vert);


        int strt, end;
        //int source_vert = 7;

        strt = clock();

        cudaProfilerStart();

        check(nvgraphSssp(handle, graph, 0,  &source_vert, 0));
        //printf("DONE!\n");

        cudaProfilerStop();

        end = clock();

        double elapsed = ((double) (end - strt)) / CLOCKS_PER_SEC;

        printf("NVGRAPH time: %f\n", elapsed);


        /*
        // Get and print result
        check(nvgraphGetVertexData(handle, graph, (void*)sssp_1_h, 0));
        //printf("DONE!\n");
        */


        //Clean 
        free(sssp_1_h); 
        free(vertex_dim);  
        free(vertex_dimT); 
        free(CSC_input);

        check(nvgraphDestroyGraphDescr(handle, graph));
        check(nvgraphDestroy(handle));
        
        free(destination_indices_h);
        free(source_offsets_h);
        free(row_ind);
        free(weights_h);
        

        return 0;
}
