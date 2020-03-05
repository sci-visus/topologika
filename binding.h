struct priority_queue;


struct priority_queue *
pq_create(void);

void
pq_destroy(struct priority_queue *pq);

void
pq_enqueue(struct priority_queue *pq, double value, int region_index, int vertex_index, int offset);

int
pq_size(struct priority_queue *pq);

void
pq_dequeue(struct priority_queue *pq, int *region_index, int *vertex_index, int *arc_index);
