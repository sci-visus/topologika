#include <assert.h>
#include <stdio.h>
#include <queue>


struct item {
	double value;
	int region_index, vertex_index;
	int arc_index;
};

static bool
operator<(struct item const &v0, struct item const &v1)
{
	if (v0.value == v1.value) {
		assert(false);
	}
	return v0.value < v1.value;
};

struct priority_queue {
	std::priority_queue<struct item> queue;
};

extern "C" struct priority_queue *
pq_create(void)
{
	return new struct priority_queue {};
}

extern "C" void
pq_destroy(struct priority_queue *pq)
{
	delete pq;
}

extern "C" int
pq_size(struct priority_queue *pq)
{
	return (int)pq->queue.size();
}

extern "C" void
pq_enqueue(struct priority_queue *pq, double value, int region_index, int vertex_index, int offset)
{
	assert(pq != NULL);
	printf("enqueue %f, r %d v %d\n", value, region_index, vertex_index);
	pq->queue.push({ value, region_index, vertex_index, offset });
}


extern "C" void
pq_dequeue(struct priority_queue *pq, int *region_index, int *vertex_index, int *arc_index)
{
	assert(!pq->queue.empty());
	struct item x = pq->queue.top();
	*region_index = x.region_index;
	*vertex_index = x.vertex_index;
	*arc_index = x.arc_index;
	pq->queue.pop();
}
