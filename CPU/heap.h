struct heap
{
	int *arr;
	int count;
	int capacity;
};
typedef struct heap Heap;

Heap* create_heap(int capacity);
void insert_to_heap(Heap *h, int key);
void heapify_up(Heap *h, int index);
void heapify_down(Heap *h, int parent);
int pop_min(Heap *h);
void print_heap(Heap *h);
bool heap_is_empty(const Heap* h);