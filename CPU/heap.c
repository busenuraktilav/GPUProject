#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

#define INF 9999

Node* create_node(int vertex_num, int distance)
{
	Node *n = (Node*)malloc(sizeof(Node*));
	n->vertex_num = vertex_num;
	n->distance = distance;

	return n;
}

Heap* create_heap(int capacity)
{
	Heap *h = (Heap *)malloc(sizeof(Heap));

	if(h == NULL)
	{
		printf("Cannot allocate heap memory!\n");
		return NULL;
	}

	h->pos = (int*)malloc(capacity * sizeof(int));
	h->count = 0;
	h->capacity = capacity;
	h->arr = (Node**)malloc(capacity * sizeof(Node));

	return h;
}


void swap_nodes(Node **a, Node **b)
{
	Node* t = *a;
	*a = *b;
	*b = t;
}


void insert_to_heap(Heap *h, int key)
{
	if(h->count < h->capacity) //heap is not full yet
	{
		h->arr[key] = create_node(key, INF);
		h->pos[key] = key;
		heapify_up(h, key);
		h->count++;
		/*
		h->arr[key]->vertex_num = key;
		h->arr[key]->distance = INF;
		heapify_up(h, key);
		h->count++;
		*/
	}
}


void heapify_up(Heap *h, int index)
{
	int temp, parent = (index-1)/2;

	if(h->arr[parent]->distance > h->arr[index]->distance)
	{
		temp = h->arr[parent]->distance;
		h->arr[parent]->distance = h->arr[index]->distance;
		h->arr[index]->distance = temp;
		heapify_up(h, parent);
	}
}


void heapify_down(Heap *h, int parent)
{
	int left, right, min, temp;

	min = parent;
	left = parent*2 + 1;
	right = parent*2 +2;

	if(left < h->count && h->arr[left]->distance < h->arr[min]->distance)
		min = left;

	if(right < h->count && h->arr[right]->distance < h->arr[min]->distance)
		min = right;

	if(min != parent)
	{
		Node *minNode = h->arr[min];
		Node *parentNode = h->arr[parent];

		//swap positions
		h->pos[minNode->vertex_num] = parent;
		h->pos[parentNode->vertex_num] = min;

		//swap nodes
		swap_nodes(&h->arr[min], &h->arr[parent]);

		heapify_down(h, min);
	}

	/*

	if(left >= h->count || left < 0)
		left = -1;

	if(right >= h->count || right < 0)
		right = -1;

	if(left != -1 && h->arr[left].distance < h->arr[parent].distance)
		min = left;
	else
		min = parent;

	if(right != -1 && h->arr[right].distance < h->arr[min].distance)
		min = right;

	if(min != parent)
	{
		temp = h->arr[min].distance;
		h->arr[min].distance = h->arr[parent].distance;
		h->arr[parent].distance = temp;

		heapify_down(h, min);
	}

	*/
}


Node* pop_min(Heap *h)
{
	if(heap_is_empty(h))
	{
		printf("Heap is empty\n");
		return NULL;
	}

	//store root node
	Node *root = h->arr[0];

	//replace root node with last node
	Node *lastNode = h->arr[h->count - 1];
	h->arr[0] = lastNode;

	//update pos of last node
	h->pos[root->vertex_num] = h->count - 1;
	h->pos[lastNode->vertex_num] = 0;

	//reduce heap size and heapify root
	h->count = h->count - 1;
	heapify_down(h, 0);

	return root;

	/*

	pop = h->arr[0].vertex_num;
	h->arr[0]  = h->arr[h->count-1];
	h->count--;
	heapify_down(h, 0);

	return pop;
	*/
	
}

void decrease_key(Heap *h, int vertex_num, int distance)
{
	//get the idex of v in heap array
	int i = h->pos[vertex_num];

	h->arr[i]->distance = distance;

	while(i && h->arr[i]->distance < h->arr[(i-1)/2]->distance)
	{
		h->pos[h->arr[i]->vertex_num] = (i-1)/2;
		h->pos[h->arr[(i-1)/2]->vertex_num] = i;
		swap_nodes(&h->arr[i], &h->arr[(i-1)/2]);

		i = (i-1)/2;
	}
}


bool is_in_heap(Heap *h, int vertex_num)
{
	if(h->pos[vertex_num] < h->count)
		return true;
	return false;
}


void print_heap(Heap *h)
{
    printf("----------Print Heap----------\n");

    for(int i = 0; i < h->count; i++){
        printf("-> %d, %d ",h->arr[i]->distance, h->arr[i]->vertex_num);
    }
    printf("->\n");
}


bool heap_is_empty(const Heap* h)
{
	return h->count == 0;
}
