#include <stdbool.h> 

struct node
{
	int vertex_num;
	int distance;
};
typedef struct node Node;

struct heap
{
	int count;
	int capacity;
	int *pos;
	Node **arr;
};
typedef struct heap Heap;

Node* create_node(int vertex_num, int distance);
Heap* create_heap(int capacity);
void swap_nodes(Node **a, Node **b);
void insert_to_heap(Heap *h, int key);
void heapify_up(Heap *h, int index);
void heapify_down(Heap *h, int parent);
Node* pop_min(Heap *h);
void decrease_key(Heap *h, int vertex_num, int distance);
bool is_in_heap(Heap *h, int vertex_num);
void print_heap(Heap *h);
bool heap_is_empty(const Heap* h);