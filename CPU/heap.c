#include <stdio.h>
#include <stdlib.h>
#include "heap.h"

Heap* create_heap(int capacity)
{
	Heap *h = (Heap *)malloc(sizeof(Heap));

	if(h == NULL)
	{
		printf("Cannot allocate heap memory!\n");
		return;
	}

	h->count = 0;
	h->capacity = capacity;
	h->arr = (int *)malloc(capacity * sizeof(int));

	return h;
}


void insert_to_heap(Heap *h, int key)
{
	if(h->count < h->capacity) //heap is not full yet
	{
		h->arr[h->count] = key;
		heapify_up(h, h->count);
		h->count++;
	}
}

void heapify_up(Heap *h, int index)
{
	int temp, parent = (index-1)/2;

	if(h->arr[parent] > h->arr[index])
	{
		temp = h->arr[parent];
		h->arr[parent] = h->arr[index];
		h->arr[index] = temp;
		heapify_up(h, parent);
	}
}


void heapify_down(Heap *h, int parent)
{
	int left, right, min, temp;

	left = parent*2 + 1;
	right = parent*2 +2;

	if(left >= h->count || left < 0)
		left = -1;

	if(right >= h->count || right < 0)
		right = -1;

	if(left != -1 && h->arr[left] < h->arr[parent])
		min = left;
	else
		min = parent;

	if(right != -1 && h->arr[right] < h->arr[min])
		min = right;

	if(min != parent)
	{
		temp = h->arr[min];
		h->arr[min] = h->arr[parent];
		h->arr[parent] = temp;

		heapify_down(h, min);
	}
}


int pop_min(Heap *h)
{
	int min_element, last_element, child, now;

	if(h->count == 0)
	{
		printf("Heap is empty\n");
		return -1;
	}

	pop = h->arr[0];
	h->arr[0] = h->arr[h->count-1];
	h->count--;
	heapify_down(h, 0);

	return pop;

	/*

	min_element = h->arr[0];
	last_element = h->arr[h->count-1];

	for(now = 0; now*2 <= (h->count); now = child)
	{
		child = now * 2;

		if(child != h->count && h->arr[child+1] < h->arr[child])
			child++;
		

		if(last_element > h->arr[child])
			h->arr[now] = h->arr[child];
		
		else
			break;
	}

	h->arr[now] = last_element;
	h->count--;

	return min_element;
	*/
}


void print(Heap *h)
{
    printf("----------Print Heap----------\n");

    for(int i = 0; i < h->count; i++){
        printf("-> %d ",h->arr[i]);
    }
    printf("->\n");
}


bool heap_is_empty(const Heap* h)
{
	return h->count == 0;
}