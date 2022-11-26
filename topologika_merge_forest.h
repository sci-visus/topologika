/*
topologika_merge_forest.h - public domain

Accelerated superlevel set analysis. Supports extraction of connected components,
finding maxima, and querying for representatives.

Requirements:
	- C99 (GCC >= 4.8, -std=gnu99, Clang >= 3.4, MSVC >= 1910)
	- OpenMP 2

References:
	- On the Locality of Persistence-Based Queries and Its Implications for Merge Forest Efficiency.
	- Toward Localized Topological Data Structures: Querying the Forest for the Tree.

Releases:
	- future
		- parallel query execution and scheduling
		- approximate analysis
		- periodic grids
	- 2020.x
		- (done) add persistence simplified queries
		- add traverse component query (allows computation during the traversal)
		- (done) support for domains that are not multiple of the region size
		- support adaptive resolution-precision grid (reduced bridge set is computed on edge stream)
	- 2019.11
		- initial port of VIS 2019 submission code with performance improvements and refactoring

Changes from the VIS code:
	- in code's simulation of simplicity, we use region_id and then vertex_id to break ties
		instead of using global row-major id
	- std::sort replaced by qsort (slower)
	- the bridge set computation does not require hardcoded neighbor region lookups
	- regions can have different sizes, thus the data dimensions do not need to be a multiple of the region size
*/

// TODO(9/12/2022): How to parallelize the components query? (split it into light traversal and vertex collection?)
// TODO(1/29/2022): I think we need to use coordinates and not indices for representing maxima/segmentation, because
//	having coordinates is very convenient for plotting the citical points (e.g., plotting maxima on images)
// TODO(1/15/2022): For global merge tree, we could reorder the segmentation (depth first order) so that we than need to find the arc, then intersection, and report subtree as a memory range. This optimization would not work for component interval queries. For merge forest, we could do the same thing locally + reorder reduced bridge set edges.
// TODO(7/19/2021): lazily extract connected components per region; the components query should return
//	only handles for each region consisting of (region_id, intersected arc_id, arc_segmentation_start, segmentation_size);
//	this well allow us to do out-of-core processing, parallel processing, preallocate memory for segmentation, streaming;
//	if we order the segmentation in depth-first order, than we can do memcpy(output, &segmentation[arc_segmentation_start], segmentation_size)
//	for each subtree (no tree traversal is needed)
// TODO(7/19/2021): support sublevel set merge tree
// TODO(2/12/2021): extract isosurface of components query
// TODO(8/3/2020): division operation for 32 bit integers is faster than 64 bit on Intel
// TODO(6/26/2020): we could store leaves in a separate array to speed up maxima query
// TODO(6/3/2020): implement custom merge sort to avoid qsort (can't pass in context)
// TODO(3/19/2020): replace divisions by shifts when region_dims is power of two (or use JIT to generate the code with a compiler to keep the code readable)
// TODO(3/19/2020): the component query has large memory overhead, we either need to use generator or return it as a compact numpy array
// TODO(3/18/2020): use next pointer (linked-list) to store the arc's children indices instead of an array (more space efficient; should have the same performance)
// TODO(3/16/2020): we could sort the todos based on the region_id in componentmax and component query to improve cache locality (inspired by distributed forest implementation)
// TODO(2/27/2020): queries should not take the domain but just forest? (thus for many queries we could do away with the data),
//	this would require caching of function values at arc's highest vertices and reduced bridge set end vertices, but we avoid a cache
//	miss since when the arc/reduced bridge set edge is accessed we also pull the values
// TODO(1/7/2020): use edges instead of vertices for the reduced bridge set computation (simplifies support of Duong's grid)
// TODO(12/3/2019): try abstract the forest component traversal so it can be reused for component_max and maxima queries
// TODO(11/25/2019): topologika_local_t change to topologika_region_t for region indexing
//	(actually, enforcing same size of both is most convenient, because we can store them in the same array
//	in the components query)
//	on the other hand, we could not do 16 bit regions
// TODO(11/19/2019): allow to save simplified field?
// TODO(11/15/2019): absan, ubsan, afl
// TODO(11/8/2019): when an out-of-memory error ocurrs, alongside the error report a lower bound on how much memory
//	would be required to build the merge forest or run the query

// NOTE(11/19/2019): compared to the old code, the reduced bridge set edges are sorted by region id
//	to break ties if local vertex ids and values are the same, thus the printout of edges per local tree
//	arc may have different order, but since the query correctness does not depend on the order
//	we get the same query output


#if !defined(TOPOLOGIKA_MERGE_FOREST_H)
#define TOPOLOGIKA_MERGE_FOREST_H


#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

// macros
#define TOPOLOGIKA_COUNT_OF(array) ((int64_t)(sizeof (array)/sizeof *(array)))

// default types
#if !defined(TOPOLOGIKA_DATA)
#define TOPOLOGIKA_DATA float
#endif

#if !defined(TOPOLOGIKA_LOCAL)
#define TOPOLOGIKA_LOCAL int32_t
#define TOPOLOGIKA_LOCAL_MAX INT32_MAX
#endif


typedef TOPOLOGIKA_DATA topologika_data_t;
typedef TOPOLOGIKA_LOCAL topologika_local_t;

struct topologika_domain;
struct topologika_merge_forest;

struct topologika_vertex {
	topologika_local_t region_index;
	topologika_local_t vertex_index;
};

struct topologika_vertex const topologika_bottom = {TOPOLOGIKA_LOCAL_MAX, TOPOLOGIKA_LOCAL_MAX};

struct topologika_pair {
	struct topologika_vertex u, s;
};

struct topologika_triplet {
	struct topologika_vertex u, s, v;
};

enum topologika_result {
	topologika_result_success,
	topologika_error_no_output,
	topologika_error_out_of_memory,
};

// 6 subdivision, matching TTK
int64_t const neighbors[][3] = {
	{0, -1, -1}, // a
	{1, -1, -1}, // b
	{0, 0, -1}, // c
	{1, 0, -1}, // d
	{0, -1, 0}, // e
	{1, -1, 0}, // f
	{1, 0, 0}, // h
	{-1, 0, 1}, // diagonal g
	{0, 0, 1}, // diagonal h
	{-1, 0, 0}, // horizontal g
	{-1, 1, 0}, // body c
	{0, 1, 0}, // body d
	{-1, 1, 1}, // body g
	{0, 1, 1}, // body h
};
/*
// 6 subdivision, matching reeber
int64_t const neighbors[][3] = {
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 0},
	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1},
	{-1, 0, 0},
	{0, -1, 0},
	{-1, -1, 0},
	{0, 0, -1},
	{-1, 0, -1},
	{0, -1, -1},
	{-1, -1, -1},
};
*/
enum {neighbor_count = TOPOLOGIKA_COUNT_OF(neighbors)};

// TODO: are these style of arrays error prone? (compared to having a malloced pointer to vertices)
struct topologika_component {
	int64_t count;
	int64_t capacity;
	struct topologika_vertex data[];
};

struct topologika_merge_tree_arc {
	// TODO: store function value?
	topologika_local_t max_vertex_id; // TODO: could read from segmentation
					  // TODO: figure out the common case, indirect for the uncommon case
	topologika_local_t children[neighbor_count]; // TODO: use linked list
	topologika_local_t child_count;
	topologika_local_t parent;

	// TODO(3/5/2021): store in a separate array?
	topologika_local_t subtree_max_id;
};



// construction
enum topologika_result
topologika_compute_merge_forest_from_grid(topologika_data_t const *data, int64_t const *data_dims, int64_t const *region_dims,
	struct topologika_domain **out_domain, struct topologika_merge_forest **out_forest, double *out_construction_time);


// queries
enum topologika_result
topologika_query_componentmax(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_vertex *out_max_vertex);

enum topologika_result
topologika_query_maxima(struct topologika_domain const *domain, struct topologika_merge_forest const *forest,
	struct topologika_vertex **out_maxima, int64_t *out_maximum_count);

// TODO: use topologika_data_t for threshold?
enum topologika_result
topologika_query_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_component **out_component);

enum topologika_result
topologika_query_components(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_component ***out_components, int64_t *out_component_count);

enum topologika_result
topologika_query_components_ids(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_vertex **out_components_ids, int64_t *out_component_count);

// TODO: how to name this query?
enum topologika_result
topologika_traverse_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc));

enum topologika_result
topologika_traverse_components(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc));


// utilities
static inline bool
topologika_vertex_eq(struct topologika_vertex v, struct topologika_vertex u)
{
	return v.region_index == u.region_index && v.vertex_index == u.vertex_index;
}


// legacy conversion functions from and to global coordinate space
struct topologika_vertex
topologika_global_index_to_vertex(int64_t const *dims, struct topologika_domain const *domain, int64_t global_vertex_index);

int64_t
topologika_vertex_to_global_index(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex vertex);








#if defined(TOPOLOGIKA_MERGE_FOREST_IMPLEMENTATION)


// compile-time configuration
bool const record_events = false;


// platform-dependent code
#if defined(_WIN64)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int64_t
topologika_usec_counter(void)
{
	LARGE_INTEGER freq = {0};
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER time = {0};
	QueryPerformanceCounter(&time);
	return 1000000LL*time.QuadPart/freq.QuadPart;
}

int64_t
topologika_atomic_add(volatile int64_t *addend, int64_t value)
{
	return _InterlockedExchangeAdd64(addend, value);
}

int64_t
topologika_thread_id(void)
{
	return GetCurrentThreadId();
}

// Linux
#elif __linux__

#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// TODO: do we need volatile?
int64_t
topologika_atomic_add(volatile int64_t *addend, int64_t value)
{
	return __sync_fetch_and_add(addend, value);
}

int64_t
topologika_usec_counter(void)
{
	struct timespec counter;
	clock_gettime(CLOCK_MONOTONIC, &counter);
	return (1000000000LL*counter.tv_sec + counter.tv_nsec)/1000LL;
}

int64_t
topologika_thread_id(void)
{
	return syscall(SYS_gettid);
}

#else
#error "Only Windows 64-bit, and Linux are supported."
#endif


static inline int64_t
topologika_max(int64_t a, int64_t b)
{
	return (a > b) ? a : b;
}


// profiling events recording
enum topologika_event_color {
	topologika_event_color_green,
	topologika_event_color_blue,
	topologika_event_color_gray,
	topologika_event_color_orange,
	topologika_event_color_black,
	topologika_event_color_white,
};

// NOTE: packed because we want recording of events to be fast (to minimize
//	memory and time impact of profiling)
// 32 bytes
struct topologika_event {
	int64_t ts;
	char name[20];
	int16_t tid;
	char ph;
	int8_t color; // use int8_t instead of enum to be more compact
};

struct topologika_events {
	int64_t count;
	int64_t capacity;
	struct topologika_event data[];
};

void
topologika_event_begin(struct topologika_events *events, enum topologika_event_color color, char const *name)
{
	if (!record_events) {
		return;
	}

	int64_t offset = topologika_atomic_add(&events->count, 1);
	assert(offset < events->capacity);

	struct topologika_event event = {
		.color = color,
		.ph = 'B',
		.ts = topologika_usec_counter(),
		.tid = (int16_t)topologika_thread_id(),
	};
	int written = snprintf(event.name, sizeof event.name, "%s", name);
	assert(written >= 0 && written < TOPOLOGIKA_COUNT_OF(event.name));
	events->data[offset] = event;
}

void
topologika_event_end(struct topologika_events *events)
{
	if (!record_events) {
		return;
	}

	int64_t offset = topologika_atomic_add(&events->count, 1);
	assert(offset < events->capacity);

	struct topologika_event event = {
		.ph = 'E',
		.ts = topologika_usec_counter(),
		.tid = (int16_t)topologika_thread_id(),
	};
	events->data[offset] = event;
}

void
topologika_write_events(char const *file_path, struct topologika_events const *events)
{
	// from Go tracing tool
	char const *color_names[] = {
		[topologika_event_color_green] = "good",
		[topologika_event_color_blue] = "vsync_highlight_color",
		[topologika_event_color_gray] = "generic_work",
		[topologika_event_color_orange] = "thread_state_iowait",
		[topologika_event_color_black] = "black",
		[topologika_event_color_white] = "white",
	};

	FILE *fp = fopen(file_path, "wb");
	assert(fp != NULL);

	fprintf(fp, "{\"traceEvents\":[");
	for (int64_t i = 0; i < events->count; i++) {
		struct topologika_event event = events->data[i];

		if (event.ph == 'B') {
			fprintf(fp, "{\"name\":\"%s\",\"cname\":\"%s\",\"ph\":\"%c\",\"pid\":0,\"tid\":%"PRIi16",\"ts\":%"PRIi64"}",
				event.name, color_names[event.color], event.ph, event.tid, event.ts);
		} else {
			fprintf(fp, "{\"ph\":\"%c\",\"pid\":0,\"tid\":%"PRIi16",\"ts\":%"PRIi64"}", event.ph, event.tid, event.ts);
		}

		if (i + 1 != events->count) {
			fprintf(fp, ",");
		}
	}
	fprintf(fp, "]}\n");

	fclose(fp);
}






enum topologika_type {
	topologika_type_uint16,
	topologika_type_float,
	topologika_type_double,
};



struct edge_vertex {
	topologika_local_t vertex_id;
	uint32_t region_id;
};
struct edge {
	struct edge_vertex lower;
	struct edge_vertex higher;
	topologika_data_t lower_value;
};

struct reduced_bridge_set_edge {
	topologika_local_t local_id;
	topologika_local_t neighbor_region_id;
	topologika_local_t neighbor_local_id;
};



struct region {
	topologika_data_t *data; // TODO(2/28/2020): union for different types?
	int64_t dims[3];
	enum topologika_type type;
};

struct topologika_domain {
	int64_t data_dims[3];
	int64_t dims[3]; // regions not whole volume; whole volume is dims[0]*regions[0].dims[0], ...
	int64_t region_dims[3];
	struct region regions[];
};




struct reduced_bridge_set {
	int64_t edge_count;
	struct reduced_bridge_set_edge edges[];
};

struct topologika_merge_tree {
	struct topologika_merge_tree_arc *arcs;
	int64_t arc_count;

	// indirect to compact segmentation representation that allows sublevel set extraction with single memcpy
	// includes the node's vertex_id
	// separate from arc as we do not require it during tree traversal
	// TODO: group offset and count into a struct?
	topologika_local_t *segmentation_offsets;
	topologika_local_t *segmentation_counts;

	topologika_local_t *segmentation;
	topologika_local_t *vertex_to_arc;

	struct reduced_bridge_set *reduced_bridge_set;
	// TODO: could these be topologika_local_t type and provably never overflow?
	int64_t *reduced_bridge_set_offsets;
	int64_t *reduced_bridge_set_counts;
};

struct topologika_merge_forest {
	int64_t merge_tree_count;
	struct topologika_merge_tree merge_trees[];
};








// stack memory allocator for scratch buffer
struct stack_allocator {
	unsigned char *memory;
	size_t offset;
	size_t size;
};

struct stack_allocation {
	void *ptr;
	size_t offset;
};

// TODO: memory passed in must be zero initialized
// TODO: memory passed in must be page aligned
void
stack_allocator_init(struct stack_allocator *allocator, size_t size, unsigned char *memory)
{
	*allocator = (struct stack_allocator){
		.memory = memory,
		.offset = 0,
		.size = size,
	};
}


struct stack_allocation
stack_allocator_alloc(struct stack_allocator *allocator, size_t alignment, size_t size)
{
	assert(allocator->memory != NULL);
	assert(alignment <= 4096);
	assert((alignment & (alignment - 1)) == 0); // check for power of two alignment as we want to avoid division in when computing the aligned offset

	size_t mask = ~(alignment - 1);
	size_t aligned_offset = (allocator->offset + alignment - 1)&mask;

	// assumes used_size <= size; the arithmetic is robust to wrapping
	if (aligned_offset + size > allocator->size || size == 0) {
		return (struct stack_allocation){.ptr = NULL};
	}

	struct stack_allocation allocation = {.offset = allocator->offset};
	allocator->offset = aligned_offset + size;
	allocation.ptr = allocator->memory + aligned_offset;
	assert((uint64_t)allocation.ptr%alignment == 0);
	return allocation;
}


void
stack_allocator_free(struct stack_allocator *allocator, struct stack_allocation allocation)
{
	allocator->offset = allocation.offset;
}




// disjoint set
// NOTE: we are doing only path compression
struct topologika_disjoint_set {
	int64_t count;
	topologika_local_t parent[];
};

struct topologika_disjoint_set_handle {
	topologika_local_t id;
};
struct topologika_disjoint_set_handle topologika_disjoint_set_invalid = {TOPOLOGIKA_LOCAL_MAX};

struct topologika_disjoint_set_handle
topologika_disjoint_set_mk(struct topologika_disjoint_set *ds, topologika_local_t id)
{
	assert(id < ds->count);
	ds->parent[id] = id;
	return (struct topologika_disjoint_set_handle){id};
}

struct topologika_disjoint_set_handle
topologika_disjoint_set_find_no_check(struct topologika_disjoint_set *ds, topologika_local_t id)
{
	if (ds->parent[id] == id) {
		return (struct topologika_disjoint_set_handle){id};
	}
	ds->parent[id] = topologika_disjoint_set_find_no_check(ds, ds->parent[id]).id;
	return (struct topologika_disjoint_set_handle){ds->parent[id]};
}

struct topologika_disjoint_set_handle
topologika_disjoint_set_find(struct topologika_disjoint_set *ds, topologika_local_t id)
{
	assert(id < ds->count);
	if (ds->parent[id] == TOPOLOGIKA_LOCAL_MAX) {
		return topologika_disjoint_set_invalid;
	}
	return topologika_disjoint_set_find_no_check(ds, id);
}

struct topologika_disjoint_set_handle
topologika_disjoint_set_union(struct topologika_disjoint_set *ds, struct topologika_disjoint_set_handle a, struct topologika_disjoint_set_handle b)
{
	ds->parent[b.id] = a.id;
	return a;
}





struct topologika_sort_vertex {
	topologika_data_t value;
	topologika_local_t index;
};

int
topologika_decreasing_cmp(void const *a, void const *b)
{
	struct topologika_sort_vertex const *v0 = b;
	struct topologika_sort_vertex const *v1 = a;

	if (v0->value == v1->value) {
		return (v0->index > v1->index) - (v0->index < v1->index);
	}
	return (v0->value < v1->value) ? -1 : 1;
}


enum topologika_result
compute_merge_tree(struct region const *region, struct stack_allocator *stack_allocator, struct topologika_events *events, int64_t const *dims,
	struct topologika_merge_tree *tree)
{
	// TODO: runtime check that we deallocate the stack allocations in a reverse order; or even
	//	simpler would be just use a linear allocator and throw everything away at the end of the function
	// TODO: we could do a temporary large allocation and then copy to malloced memory; it is more robust with
	//	fewer failure points, but readibility suffers

	*tree = (struct topologika_merge_tree){0};
	struct stack_allocator initial_allocator = *stack_allocator;

	int64_t vertex_count = dims[0]*dims[1]*dims[2];

	// TODO: return pointer and then when we free data pass size
	struct topologika_sort_vertex *vertices = NULL;
	struct stack_allocation vertices_allocation = stack_allocator_alloc(stack_allocator, 8, vertex_count*sizeof *vertices);
	assert(vertices_allocation.ptr != NULL);
	vertices = vertices_allocation.ptr;

	int64_t arc_capacity = 32;
	{
		tree->arcs = malloc(arc_capacity*sizeof *tree->arcs);
		tree->segmentation_counts = malloc(arc_capacity*sizeof *tree->segmentation_counts);
		if (tree->arcs == NULL || tree->segmentation_counts == NULL) {
			goto out_of_memory;
		}
	}

	// inlining the data reduces number of indirections in the sort
	topologika_event_begin(events, topologika_event_color_green, "Init");
#if !defined(TOPOLOGIKA_THRESHOLD)
	for (int64_t i = 0; i < vertex_count; i++) {
		vertices[i] = (struct topologika_sort_vertex){.value = region->data[i], .index = (topologika_local_t)i};
	}
#else
	int64_t thresholded_count = 0;
	for (int64_t i = 0; i < vertex_count; i++) {
		if (region->data[i] >= TOPOLOGIKA_THRESHOLD) {
			vertices[thresholded_count++] = (struct topologika_sort_vertex){.value = region->data[i], .index = (topologika_local_t)i};
		}
	}
	vertex_count = thresholded_count;
#endif
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Sort");
	qsort(vertices, vertex_count, sizeof *vertices, topologika_decreasing_cmp);
	topologika_event_end(events);

	int64_t vertex_count_ = dims[0]*dims[1]*dims[2];
	struct topologika_disjoint_set *components = NULL;
	struct stack_allocation components_allocation = stack_allocator_alloc(stack_allocator, 8, sizeof *components + vertex_count*sizeof *components->parent);
	if (components_allocation.ptr == NULL) {
		// TODO: free other memory
		return topologika_error_out_of_memory;
	}
	components = components_allocation.ptr;
	components->count = vertex_count;

	topologika_event_begin(events, topologika_event_color_green, "Alloc");
	tree->vertex_to_arc = malloc(vertex_count_*sizeof *tree->vertex_to_arc);
	if (tree->vertex_to_arc == NULL) {
		goto out_of_memory;
	}
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Clear");
	for (int64_t i = 0; i < vertex_count_; i++) {
		tree->vertex_to_arc[i] = TOPOLOGIKA_LOCAL_MAX;
	}
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Sweep");
	for (int64_t i = 0; i < vertex_count; i++) {
		topologika_local_t vertex_idx = vertices[i].index;
		int64_t vertex_position[] = {
			vertex_idx%dims[0],
			vertex_idx/dims[0]%dims[1],
			vertex_idx/(dims[0]*dims[1]),
		};

		struct topologika_disjoint_set_handle vertex_component = topologika_disjoint_set_invalid;

		// we exploit the fact that all children for a node are obtained in sequence, and thus
		//	can be appended in one go to the children buffer
		topologika_local_t children[neighbor_count];
		uint32_t child_count = 0;
		for (int64_t neighbor_i = 0; neighbor_i < neighbor_count; neighbor_i++) {
			int64_t neighbor_position[] = {
				vertex_position[0] + neighbors[neighbor_i][0],
				vertex_position[1] + neighbors[neighbor_i][1],
				vertex_position[2] + neighbors[neighbor_i][2],
			};

			// TODO: check if the compiler can do this cast optimization (e.g., fold x < 0 || x >= region->dims[0] => (uint64_t)x >= region->dims[0]
			if ((uint64_t)neighbor_position[0] >= (uint64_t)dims[0] || (uint64_t)neighbor_position[1] >= (uint64_t)dims[1] || (uint64_t)neighbor_position[2] >= (uint64_t)dims[2]) {
				continue;
			}

			int64_t neighbor_idx = neighbor_position[0] + neighbor_position[1]*dims[0] + neighbor_position[2]*dims[0]*dims[1];

			topologika_local_t neighbor_arc = tree->vertex_to_arc[neighbor_idx];
			if (neighbor_arc == TOPOLOGIKA_LOCAL_MAX) {
				continue;
			}

			struct topologika_disjoint_set_handle neighbor_component = topologika_disjoint_set_find(components, neighbor_arc);
			assert(neighbor_component.id != topologika_disjoint_set_invalid.id);

			if (vertex_component.id != neighbor_component.id) {
				children[child_count++] = neighbor_component.id; // same as the lowest arc in component

				// NOTE: assumes we get out neighbor_component label and thus regular vertex code below will work correctly
				if (vertex_component.id == topologika_disjoint_set_invalid.id) {
					vertex_component = neighbor_component;
				} else {
					// we always want lower component to point to higher component to get correct
					//	arc if neighbor was a regular vertex
					vertex_component = topologika_disjoint_set_union(components, neighbor_component, vertex_component);
				}
			}
		}

		// ensure we have enough space to store next critical point
		if (child_count != 1 && tree->arc_count == arc_capacity) {
			arc_capacity *= 2;
			struct topologika_merge_tree_arc* arcs = realloc(tree->arcs, arc_capacity * sizeof * tree->arcs);
			topologika_local_t* segmentation_counts = realloc(tree->segmentation_counts, arc_capacity * sizeof * tree->segmentation_counts);
			if (arcs == NULL || segmentation_counts == NULL) {
				goto out_of_memory;
			}
			tree->arcs = arcs;
			tree->segmentation_counts = segmentation_counts;
		}

		// maximum vertex
		if (child_count == 0) {
			tree->arcs[tree->arc_count] = (struct topologika_merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
			};
			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;
			topologika_disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			tree->arc_count++;

		// regular vertex
		} else if (child_count == 1) {
			tree->vertex_to_arc[vertex_idx] = children[0];
			tree->segmentation_counts[children[0]]++;

		// merge saddle vertex
		} else {
			tree->arcs[tree->arc_count] = (struct topologika_merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
				.child_count = child_count,
			};

			assert(child_count <= TOPOLOGIKA_COUNT_OF(tree->arcs[0].children));
			for (int64_t child_i = 0; child_i < child_count; child_i++) {
				tree->arcs[tree->arc_count].children[child_i] = children[child_i];
				tree->arcs[children[child_i]].parent = (topologika_local_t)tree->arc_count;
			}

			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;

			// disjoint set per arc
			struct topologika_disjoint_set_handle component = topologika_disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			topologika_disjoint_set_union(components, component, vertex_component); // NOTE: I think the order matters here
			tree->arc_count++;
		}
	}
	topologika_event_end(events);

	stack_allocator_free(stack_allocator, components_allocation);


	// build arc segmentation
	tree->segmentation = malloc(vertex_count*sizeof *tree->segmentation);
	if (tree->segmentation == NULL) {
		goto out_of_memory;
	}

	tree->segmentation_offsets = malloc(tree->arc_count*sizeof *tree->segmentation_offsets);
	if (tree->segmentation_offsets == NULL) {
		goto out_of_memory;
	}

	// prefix sum to figure out offsets into segmentation buffer (DFS? starting from root)
	topologika_event_begin(events, topologika_event_color_green, "Seg. offsets");
	for (int64_t i = tree->arc_count - 1, offset = 0; i >= 0; i--) {
		tree->segmentation_offsets[i] = (topologika_local_t)offset;
		offset += tree->segmentation_counts[i];
	}
	topologika_event_end(events);

	// TODO: we could do a depth-first traversal so the segmentation is linearized and any component
	//	of a subtree can be extracted with a memcpy
	// TODO: do we need the arc segmentation sorted? (now it is)
	// zero out so we can use it as offset in the assignment pass
	topologika_event_begin(events, topologika_event_color_green, "Seg. assign");
	memset(tree->segmentation_counts, 0x00, tree->arc_count*sizeof *tree->segmentation_counts);
	for (int64_t i = 0; i < vertex_count; i++) {
		topologika_local_t vertex_idx = vertices[i].index;
		topologika_local_t arc_id = tree->vertex_to_arc[vertex_idx];
		tree->segmentation[tree->segmentation_offsets[arc_id] + tree->segmentation_counts[arc_id]] = vertex_idx;
		tree->segmentation_counts[arc_id]++;
	}
	topologika_event_end(events);

	stack_allocator_free(stack_allocator, vertices_allocation);
	return topologika_result_success;

out_of_memory:
	free(tree->arcs);
	free(tree->segmentation);
	free(tree->segmentation_counts);
	free(tree->segmentation_offsets);
	free(tree->vertex_to_arc);
	*stack_allocator = initial_allocator;
	// TODO: more descriptive error message with location of code where allocation failed?
	return topologika_error_out_of_memory;
}



// the original code to compute reduced bridge set, mostly to test the cleanup research code produces the same result
// hash table
struct table {
	uint64_t count;
	uint64_t capacity;
	// TODO(2/10/2019): check if grouping keys and values is better for cache lines
	uint64_t *keys;
	uint64_t *values;
};

void
table_init(struct table *table, uint64_t capacity)//, unsigned char *memory)
{
	assert(((capacity - 1) & capacity) == 0);
	table->count = 0;
	table->capacity = capacity;
	// TODO: pass in the memory or allocator
	table->keys = malloc(2*capacity*sizeof *table->keys);
	assert(table->keys != NULL);
	memset(table->keys, 0xFF, capacity*sizeof *table->keys);
	table->values = &table->keys[capacity];
}

void
table_destroy(struct table *table)
{
	free(table->keys);
}


// from murmur 3 (fmix64)
uint64_t
table_hash(uint64_t key)
{
	key ^= key >> 33;
	key *= 0xFF51AFD7ED558CCDull;
	key ^= key >> 33;
	key *= 0xc4CEB9FE1A85EC53ull;
	key ^= key >> 33;
	return key;
}

bool
table_get(struct table const *table, uint64_t key, uint64_t *value)
{
	uint64_t mask = table->capacity - 1;
	uint64_t hash = table_hash(key);
	uint64_t idx = hash&mask;

	while (true) {
		if (table->keys[idx] == UINT64_MAX) {
			return false;
		}
		if (table->keys[idx] == key) {
			*value = table->values[idx];
			return true;
		}

		idx = (idx + 1)&mask;
	}
}

void
table_set(struct table *table, uint64_t key, uint64_t value)
{
	uint64_t mask = table->capacity - 1;
	uint64_t hash = table_hash(key);
	uint64_t idx = hash&mask;

	while (true) {
		if (table->keys[idx] == UINT64_MAX) {
			assert(table->count < table->capacity);
			table->keys[idx] = key;
			table->values[idx] = value;
			table->count++;
			return;
		} else if (table->keys[idx] == key) {
			table->values[idx] = value;
			return;
		}

		idx = (idx + 1)&mask;
	}
}


struct ds {
	struct table table;
};

struct ds_handle {uint64_t id;};

void
ds_init(struct ds *ds, uint64_t count)//, unsigned char *memory)
{
	assert(((count - 1) & count) == 0);
	// TODO: pass in allocator
	table_init(&ds->table, 4*count);//, memory);
}

void
ds_destroy(struct ds *ds)
{
	table_destroy(&ds->table);
}

struct ds_handle
ds_mk(struct ds *ds, uint64_t id)
{
#if !defined(NDEBUG)
	uint64_t value;
	assert(!table_get(&ds->table, id, &value));
#endif
	table_set(&ds->table, id, id);
	return (struct ds_handle){id};
}

struct ds_handle
ds_find_no_check(struct ds *ds, uint64_t id)
{
	uint64_t value;
	table_get(&ds->table, id, &value);
	if (id == value) {
		return (struct ds_handle){id};
	}

	struct ds_handle result = ds_find_no_check(ds, value);
	table_set(&ds->table, id, result.id);
	return result;
}

struct ds_handle
ds_find(struct ds *ds, uint64_t id)
{
	uint64_t value;
	if (!table_get(&ds->table, id, &value)) {
		return (struct ds_handle){UINT64_MAX};
	}

	return ds_find_no_check(ds, id);
}

struct ds_handle
ds_union(struct ds *ds, struct ds_handle a, struct ds_handle b)
{
	uint64_t value;
	table_get(&ds->table, a.id, &value);
	table_set(&ds->table, b.id, value);
	return a;

}


struct vertex {
	topologika_data_t value;
	int64_t global_id;
};

struct bedge {
	uint64_t global_id;
	uint64_t neighbor_global_id;
};


int64_t
topologika_local_to_global_id(struct topologika_domain const *domain, int64_t vertex_id, int64_t region_id)
{
	int64_t corner[] = {
		domain->region_dims[0]*(region_id%domain->dims[0]),
		domain->region_dims[1]*(region_id/domain->dims[0]%domain->dims[1]),
		domain->region_dims[2]*(region_id/(domain->dims[0]*domain->dims[1])),
	};
	int64_t position[] = {
		corner[0] + (vertex_id%domain->regions[region_id].dims[0]),
		corner[1] + (vertex_id/domain->regions[region_id].dims[0]%domain->regions[region_id].dims[1]),
		corner[2] + (vertex_id/(domain->regions[region_id].dims[0]*domain->regions[region_id].dims[1])),
	};

	return position[0] + domain->data_dims[0]*(position[1] + domain->data_dims[1]*position[2]);
}

int64_t
global_position_to_region_id(struct topologika_domain const *domain, int64_t const *global_position)
{
	int64_t region_position[] = {
		global_position[0]/domain->region_dims[0],
		global_position[1]/domain->region_dims[1],
		global_position[2]/domain->region_dims[2],
	};
	return region_position[0] + domain->dims[0]*(region_position[1] + domain->dims[1]*region_position[2]);
}


topologika_local_t
global_position_to_local_id(struct topologika_domain const *domain, int64_t region_id, int64_t const *global_position)
{
	topologika_local_t local_position[] = {
		(topologika_local_t)(global_position[0]%domain->region_dims[0]),
		(topologika_local_t)(global_position[1]%domain->region_dims[1]),
		(topologika_local_t)(global_position[2]%domain->region_dims[2]),
	};
	return (topologika_local_t)(local_position[0] + domain->regions[region_id].dims[0]*(local_position[1] +  domain->regions[region_id].dims[1]*local_position[2]));
}



void
radix_sort_bs(int64_t count, struct vertex *in, struct vertex *out, int shift)
{
	int64_t index[256] = {0};
	for (int64_t i = 0; i < count; i++) {
		union cast {
			topologika_data_t in;
			uint32_t out;
		};
		uint32_t x = (union cast){.in = in[i].value}.out;
		uint32_t tmp = x^(((int32_t)x >> 31) | 0x80000000);
		index[(tmp >> shift) & 0xFF]++;
	}
	for (int64_t i = 0, offset = 0; i < 256; i++) {
		offset += index[i];
		index[i] = offset - index[i];
	}
	for (int64_t i = 0; i < count; i++) {
		union cast {
			topologika_data_t in;
			uint32_t out;
		};
		uint32_t x = (union cast){.in = in[i].value}.out;
		uint32_t tmp = x^(((int32_t)x >> 31) | 0x80000000);
		out[index[(tmp >> shift) & 0xFF]++] = in[i];
	}
}

void
sort_reduced_bridge_set(int64_t count, struct vertex *vertices, struct vertex *tmp)
{
	radix_sort_bs(count, vertices, tmp, 0);
	radix_sort_bs(count, tmp, vertices, 8);
	radix_sort_bs(count, vertices, tmp, 16);
	radix_sort_bs(count, tmp, vertices, 24);
}


int64_t
topologika_next_power(int64_t n)
{
	assert(n >= 0);
	int64_t next = 1;
	while (next < n) {
		next *= 2;
	}
	return next;
}

struct reduced_bridge_set *
compute_reduced_bridge_set_internal(struct topologika_domain const *domain, int64_t region_id, int64_t neighbor_region_id,
	int64_t vertex_count, struct vertex *vertices,
	int64_t neighbors_local_count, int64_t (*const neighbors_local)[3],
	int64_t neighbors_outside_count,  int64_t (*const neighbors_outside)[3],
	struct stack_allocator *stack_allocator, struct topologika_events *events)
{
	if (vertex_count == 0) {
		return NULL;
	}

#if defined(TOPOLOGIKA_DUMP)
	printf("before sort\n");
	for (int64_t i = 0; i < vertex_count; i++) {
		struct vertex vertex = vertices[i];
		printf("%f %d\n", vertex.value, vertex.global_id);
	}
	printf("done\n");
#endif

	// sort vertices
	topologika_event_begin(events, topologika_event_color_orange, "Sort");
	struct stack_allocation tmp_allocation = stack_allocator_alloc(stack_allocator, 8, vertex_count*sizeof *vertices);
	sort_reduced_bridge_set(vertex_count, vertices, tmp_allocation.ptr);
	stack_allocator_free(stack_allocator, tmp_allocation);
	topologika_event_end(events);

#if defined(TOPOLOGIKA_DUMP)
	printf("after sort\n");
	for (int64_t i = 0; i < vertex_count; i++) {
		struct vertex vertex = vertices[i];
		printf("%f %d\n", vertex.value, vertex.global_id);
	}
	printf("done\n");
#endif

	int64_t const max_region_dim = topologika_max(domain->region_dims[0], topologika_max(domain->region_dims[1], domain->region_dims[2])); // NOTE: or use region's dims domain->regions[region_id].dims
	int64_t const bridge_set_capacity = max_region_dim*max_region_dim*8; // TODO: works only for 14 or smaller neighborhood
	uint32_t bridge_set_count = 0;
	struct bedge *bridge_set = NULL;
	struct stack_allocation bridge_set_allocation = stack_allocator_alloc(stack_allocator, 8, bridge_set_capacity*sizeof *bridge_set);
	bridge_set = bridge_set_allocation.ptr;
	assert(bridge_set != NULL);
	assert(bridge_set_capacity < TOPOLOGIKA_LOCAL_MAX); // TODO: we are using TOPOLOGIKA_LOCAL_MAX as invalid component id for now

	// TODO: scratch buffer (stack allocator)
	struct ds components;
	// TODO(2/28/2020): allocate the max of region_dims[0], region_dims[1], region_dims[2] for regions with
	//	non-uniform aspect radio (e.g., 64x32x32); we could also pass the correct face size from
	//	the compute_reduced_bridge_set function
	int64_t count = topologika_next_power(max_region_dim*max_region_dim);
	ds_init(&components, count/*vertex_count*/);

	// descending
	topologika_event_begin(events, topologika_event_color_orange, "Sweep");
#if defined(TOPOLOGIKA_DUMP)
	printf("sorted input to bridge set for region %d, neighbor %d\n", region_id, neighbor_region_id);
#endif
	for (int64_t i = vertex_count - 1; i >= 0; i--) {
		struct vertex vertex = vertices[i];
#if defined(TOPOLOGIKA_DUMP)
		printf("%f %d\n", vertex.value, vertex.global_id);
#endif
		uint64_t vertex_position[] = {
			vertex.global_id%domain->data_dims[0],
			vertex.global_id/domain->data_dims[0]%domain->data_dims[1],
			vertex.global_id/(domain->data_dims[0]*domain->data_dims[1]),
		};

		struct ds_handle vertex_component = ds_mk(&components, vertex.global_id);

		// first process region local neighbors
		for (uint32_t neighbor_i = 0; neighbor_i < neighbors_local_count; neighbor_i++) {
			uint64_t neighbor_position[] = {
				vertex_position[0] + neighbors_local[neighbor_i][0],
				vertex_position[1] + neighbors_local[neighbor_i][1],
				vertex_position[2] + neighbors_local[neighbor_i][2],
			};

			// TODO: compiler probably can optimize < && >= -> uint64_t >=
			if ((uint64_t)neighbor_position[0] >= (uint64_t)domain->data_dims[0] || (uint64_t)neighbor_position[1] >= (uint64_t)domain->data_dims[1] || (uint64_t)neighbor_position[2] >= (uint64_t)domain->data_dims[2]) {
				continue;
			}

			// TODO: we could precompute an offset table for each neighbor
			uint64_t neighbor_id = neighbor_position[0] + domain->data_dims[0]*(neighbor_position[1] + domain->data_dims[1]*neighbor_position[2]);

			struct ds_handle neighbor_component = ds_find(&components, neighbor_id);
			if (neighbor_component.id == UINT64_MAX) {
				continue;
			}
			if (vertex_component.id == neighbor_component.id) {
				continue;
			}

			vertex_component = ds_union(&components, vertex_component, neighbor_component);
		}

		// second process outside neighbors
		for (uint32_t neighbor_i = 0; neighbor_i < neighbors_outside_count; neighbor_i++) {
			uint64_t neighbor_position[] = {
				vertex_position[0] + neighbors_outside[neighbor_i][0],
				vertex_position[1] + neighbors_outside[neighbor_i][1],
				vertex_position[2] + neighbors_outside[neighbor_i][2],
			};

			// TODO: compiler probably can optimize < && >= -> uint64_t >=
			if ((uint64_t)neighbor_position[0] >= (uint64_t)domain->data_dims[0] || (uint64_t)neighbor_position[1] >= (uint64_t)domain->data_dims[1] || (uint64_t)neighbor_position[2] >= (uint64_t)domain->data_dims[2]) {
				continue;
			}

			// TODO: we could precompute an offset table for each neighbor
			uint64_t neighbor_id = neighbor_position[0] + domain->data_dims[0]*(neighbor_position[1] + domain->data_dims[1]*neighbor_position[2]);

			struct ds_handle neighbor_component = ds_find(&components, neighbor_id);
			if (neighbor_component.id == UINT64_MAX) {
				continue;
			}
			if (vertex_component.id == neighbor_component.id) {
				continue;
			}

			vertex_component = ds_union(&components, vertex_component, neighbor_component);

			assert(bridge_set_count < bridge_set_capacity);
			bridge_set[bridge_set_count++] = (struct bedge){
				.global_id = vertex.global_id,
				.neighbor_global_id = neighbor_id,
			};
		}
	}
	topologika_event_end(events);
	ds_destroy(&components);

	// convert global id to local id and neighbor (region + local) id
	// TODO: optimize
	struct reduced_bridge_set *reduced_bridge_set = malloc(sizeof *reduced_bridge_set + bridge_set_count*sizeof *reduced_bridge_set->edges);
	assert(reduced_bridge_set != NULL);
	reduced_bridge_set->edge_count = 0;
	{
		for (uint32_t i = 0; i < bridge_set_count; i++) {
			struct bedge edge = bridge_set[i];

			int64_t position[] = {
				edge.global_id%domain->data_dims[0],
				edge.global_id/domain->data_dims[0]%domain->data_dims[1],
				edge.global_id/(domain->data_dims[0]*domain->data_dims[1]),
			};
			int64_t v0_region_id = global_position_to_region_id(domain, position);


			if (v0_region_id == region_id) {
				topologika_local_t v0_local_id = global_position_to_local_id(domain, region_id, position);

				int64_t neighbor_position[] = {
					edge.neighbor_global_id%domain->data_dims[0],
					edge.neighbor_global_id/domain->data_dims[0]%domain->data_dims[1],
					edge.neighbor_global_id/(domain->data_dims[0]*domain->data_dims[1]),
				};
				topologika_local_t v1_local_id = global_position_to_local_id(domain, neighbor_region_id, neighbor_position);

				reduced_bridge_set->edges[reduced_bridge_set->edge_count++] = (struct reduced_bridge_set_edge){
					.local_id = v0_local_id,
					.neighbor_region_id = (uint32_t)neighbor_region_id,
					.neighbor_local_id = v1_local_id,
				};
			} else {
				topologika_local_t v0_local_id = global_position_to_local_id(domain, neighbor_region_id, position);

				int64_t neighbor_position[] = {
					edge.neighbor_global_id%domain->data_dims[0],
					edge.neighbor_global_id/domain->data_dims[0]%domain->data_dims[1],
					edge.neighbor_global_id/(domain->data_dims[0]*domain->data_dims[1]),
				};
				topologika_local_t v1_local_id = global_position_to_local_id(domain, region_id, neighbor_position);

				reduced_bridge_set->edges[reduced_bridge_set->edge_count++] = (struct reduced_bridge_set_edge){
					.local_id = v1_local_id,
					.neighbor_region_id = (uint32_t)neighbor_region_id,
					.neighbor_local_id = v0_local_id,
				};
			}
		}
	}
	stack_allocator_free(stack_allocator, bridge_set_allocation);

	return reduced_bridge_set;

}

int64_t
compute_reduced_bridge_set_vertices(struct topologika_domain const *domain, int64_t region_id, int64_t const *neighbors, struct vertex *vertices, int64_t vertex_count)
{
	struct region const *region = &domain->regions[region_id];

	int64_t start_i = (neighbors[0] == 1) ? region->dims[0] - 1 : 0;
	int64_t end_i = (neighbors[0] != 0) ? (start_i + 1) : region->dims[0];
	int64_t start_j = (neighbors[1] == 1) ? region->dims[1] - 1 : 0;
	int64_t end_j = (neighbors[1] != 0) ? (start_j + 1) : region->dims[1];
	int64_t start_k = (neighbors[2] == 1) ? region->dims[2] - 1 : 0;
	int64_t end_k = (neighbors[2] != 0) ? (start_k + 1) : region->dims[2];

	for (int64_t k = start_k; k < end_k; k++) {
		for (int64_t j = start_j; j < end_j; j++) {
			for (int64_t i = start_i; i < end_i; i++) {
				topologika_local_t id = (topologika_local_t)(i + region->dims[0]*(j + region->dims[1]*k));
#if !defined(TOPOLOGIKA_THRESHOLD)
				vertices[vertex_count++] = (struct vertex){region->data[id], topologika_local_to_global_id(domain, id, region_id)};
#else
				if (region->data[id] >= TOPOLOGIKA_THRESHOLD) {
					vertices[vertex_count++] = (struct vertex){region->data[id], topologika_local_to_global_id(domain, id, region_id)};
				}
#endif
			}
		}
	}

	return vertex_count;
}

enum topologika_result
compute_reduced_bridge_set(struct topologika_domain const *domain, int64_t region_id, struct stack_allocator *stack_allocator, struct topologika_events *events,
	struct reduced_bridge_set **out_reduced_bridge_set)
{
	int64_t region_position[] = {
		region_id%domain->dims[0],
		region_id/domain->dims[0]%domain->dims[1],
		region_id/(domain->dims[0]*domain->dims[1]),
	};

	struct vertex *vertices = NULL;
	int64_t max_region_dim = topologika_max(domain->region_dims[0], topologika_max(domain->region_dims[1], domain->region_dims[2]));
	struct stack_allocation vertices_allocation = stack_allocator_alloc(stack_allocator, 8, 2*max_region_dim*max_region_dim*sizeof *vertices);
	vertices = vertices_allocation.ptr;
	assert(vertices != NULL);
	struct reduced_bridge_set *reduced_bridge_sets[neighbor_count] = {0};

	for (int64_t neighbor_index = 0; neighbor_index < neighbor_count; neighbor_index++) {
		int64_t neighbor_region_position[] = {
			region_position[0] + neighbors[neighbor_index][0],
			region_position[1] + neighbors[neighbor_index][1],
			region_position[2] + neighbors[neighbor_index][2],
		};

		if (neighbor_region_position[0] < 0 || neighbor_region_position[0] >= domain->dims[0] ||
			neighbor_region_position[1] < 0 || neighbor_region_position[1] >= domain->dims[1] ||
			neighbor_region_position[2] < 0 || neighbor_region_position[2] >= domain->dims[2]) {
			continue;
		}

		// build inside and outside neighborhoods
		int64_t neighbors_inside[neighbor_count][3];
		int64_t neighbors_outside[neighbor_count][3];

		int64_t neighbor_inside_count = 0;
		int64_t neighbor_outside_count = 0;
		for (int64_t i = 0; i < neighbor_count; i++) {
			if ((neighbors[neighbor_index][0] != 0 && neighbors[i][0] != 0) ||
				(neighbors[neighbor_index][1] != 0 && neighbors[i][1] != 0) ||
				(neighbors[neighbor_index][2] != 0 && neighbors[i][2] != 0)) {

				neighbors_outside[neighbor_outside_count][0] = neighbors[i][0];
				neighbors_outside[neighbor_outside_count][1] = neighbors[i][1];
				neighbors_outside[neighbor_outside_count][2] = neighbors[i][2];
				neighbor_outside_count++;
			} else {
				neighbors_inside[neighbor_inside_count][0] = neighbors[i][0];
				neighbors_inside[neighbor_inside_count][1] = neighbors[i][1];
				neighbors_inside[neighbor_inside_count][2] = neighbors[i][2];
				neighbor_inside_count++;
			}
		}

		int64_t vertex_count = 0;
		int64_t neighbor_region_id = neighbor_region_position[0] +
			neighbor_region_position[1]*domain->dims[0] +
			neighbor_region_position[2]*domain->dims[0]*domain->dims[1];


		// simulation of simplicity is on region_id first, so lower region must be added to the vertices
		//	buffer first too (we use stable sort); inside the region the simulation of simlicity is dictated by row-major order
		if (region_id < neighbor_region_id) {
			vertex_count = compute_reduced_bridge_set_vertices(domain, region_id, neighbors[neighbor_index], vertices, vertex_count);

			int64_t flipped_neighbors[] = {-neighbors[neighbor_index][0], -neighbors[neighbor_index][1], -neighbors[neighbor_index][2]};
			vertex_count = compute_reduced_bridge_set_vertices(domain, neighbor_region_id, flipped_neighbors, vertices, vertex_count);
		} else {
			int64_t flipped_neighbors[] = {-neighbors[neighbor_index][0], -neighbors[neighbor_index][1], -neighbors[neighbor_index][2]};
			vertex_count = compute_reduced_bridge_set_vertices(domain, neighbor_region_id, flipped_neighbors, vertices, vertex_count);

			vertex_count = compute_reduced_bridge_set_vertices(domain, region_id, neighbors[neighbor_index], vertices, vertex_count);
		}

		reduced_bridge_sets[neighbor_index] = compute_reduced_bridge_set_internal(domain, region_id, neighbor_region_id, vertex_count, vertices, neighbor_inside_count, neighbors_inside, neighbor_outside_count, neighbors_outside, stack_allocator, events);
	}

	// union bridge sets
	// TODO(2/11/2019): keep per face reduced bridge set?
	struct reduced_bridge_set *reduced_bridge_set = NULL;
	{
		int64_t count = 0;
		for (int64_t i = 0; i < TOPOLOGIKA_COUNT_OF(reduced_bridge_sets); i++) {
			if (reduced_bridge_sets[i] != NULL) {
				count += reduced_bridge_sets[i]->edge_count;
			}
		}

		reduced_bridge_set = malloc(sizeof *reduced_bridge_set + count*sizeof *reduced_bridge_set->edges);
		assert(reduced_bridge_set != NULL);
		reduced_bridge_set->edge_count = 0;

		for (int64_t i = 0; i < TOPOLOGIKA_COUNT_OF(reduced_bridge_sets); i++) {
			if (reduced_bridge_sets[i] == NULL) {
				continue;
			}

			memcpy(&reduced_bridge_set->edges[reduced_bridge_set->edge_count], reduced_bridge_sets[i]->edges, reduced_bridge_sets[i]->edge_count*sizeof *reduced_bridge_sets[i]->edges);
			reduced_bridge_set->edge_count += reduced_bridge_sets[i]->edge_count;

			free(reduced_bridge_sets[i]);
		}
	}

	*out_reduced_bridge_set = reduced_bridge_set;
	stack_allocator_free(stack_allocator, vertices_allocation);
	return topologika_result_success;
}


// lowest arc id first
void
radix_sort_edges(int64_t count, struct reduced_bridge_set_edge *in, struct reduced_bridge_set_edge *out, int shift, topologika_local_t const *vertex_to_arc)
{
	int64_t index[256] = {0};
	for (int64_t i = 0; i < count; i++) {
		uint32_t tmp = vertex_to_arc[in[i].local_id];
		index[(tmp >> shift) & 0xFF]++;
	}
	for (int64_t i = 0, offset = 0; i < 256; i++) {
		offset += index[i];
		index[i] = offset - index[i];
	}
	for (int64_t i = 0; i < count; i++) {
		uint32_t tmp = vertex_to_arc[in[i].local_id];
		out[index[(tmp >> shift) & 0xFF]++] = in[i];
	}
}
void
sort_edges(int64_t count, struct reduced_bridge_set_edge *in, struct reduced_bridge_set_edge *tmp, topologika_local_t const *vertex_to_arc)
{
	assert(sizeof *vertex_to_arc == 4);
	radix_sort_edges(count, in, tmp, 0, vertex_to_arc);
	radix_sort_edges(count, tmp, in, 8, vertex_to_arc);
	radix_sort_edges(count, in, tmp, 16, vertex_to_arc);
	radix_sort_edges(count, tmp, in, 24, vertex_to_arc);
}


enum topologika_result
topologika_compute_merge_forest_from_grid(topologika_data_t const *data, int64_t const *data_dims, int64_t const *region_dims,
	struct topologika_domain **out_domain, struct topologika_merge_forest **out_forest, double *out_construction_time)
{
	// TODO: we assume region_id <= 32 bits and vertex_id <= 32 bits
	assert(region_dims[0]*region_dims[1]*region_dims[2] <= ((int64_t)1 << (8*sizeof (topologika_local_t))));

	assert(region_dims[0] <= data_dims[0] && region_dims[1] <= data_dims[1] && region_dims[2] <= data_dims[2]);

	// heap-allocated toplevel pointers
	unsigned char *stack_allocators_memory = NULL;
	struct topologika_domain *domain = NULL;
	struct topologika_merge_forest *forest = NULL;
	struct topologika_events *events = NULL;
	struct stack_allocator *stack_allocators = NULL;

	// rounded up numbers of regions in each axis
	int64_t dims[] = {
		(data_dims[0] + region_dims[0] - 1)/region_dims[0],
		(data_dims[1] + region_dims[1] - 1)/region_dims[1],
		(data_dims[2] + region_dims[2] - 1)/region_dims[2],
	};
	int64_t region_count = dims[0]*dims[1]*dims[2];

	domain = malloc(sizeof *domain + region_count*sizeof *domain->regions);
	if (domain == NULL) {
		goto out_of_memory;
	}
	*domain = (struct topologika_domain){
		.data_dims = {data_dims[0], data_dims[1], data_dims[2]},
		.dims = {dims[0], dims[1], dims[2]},
		.region_dims = {region_dims[0], region_dims[1], region_dims[2]},
	};

	// decompose row-major array into regions
	int64_t region_index = 0;
	for (int64_t k = 0; k < data_dims[2]; k += region_dims[2]) {
		for (int64_t j = 0; j < data_dims[1]; j += region_dims[1]) {
			for (int64_t i = 0; i < data_dims[0]; i += region_dims[0]) {
				int64_t rdims[] = {
					(i + region_dims[0] > data_dims[0]) ? data_dims[0] - i : region_dims[0],
					(j + region_dims[1] > data_dims[1]) ? data_dims[1] - j : region_dims[1],
					(k + region_dims[2] > data_dims[2]) ? data_dims[2] - k : region_dims[2],
				};

				domain->regions[region_index] = (struct region){
					.data = malloc(rdims[0]*rdims[1]*rdims[2]*sizeof *domain->regions[region_index].data),
					.dims = {rdims[0], rdims[1], rdims[2]},
					.type = topologika_type_float,
				};
				if (domain->regions[region_index].data == NULL) {
					goto out_of_memory;
				}

				// copy the subset of global grid into the region
				int64_t position[] = {i, j, k};
				int64_t offset = 0;
				for (int64_t k = 0; k < rdims[2]; k++) {
					for (int64_t j = 0; j < rdims[1]; j++) {
						for (int64_t i = 0; i < rdims[0]; i++) {
							int64_t index = (i + position[0]) +
								(j + position[1])*data_dims[0] +
								(k + position[2])*data_dims[0]*data_dims[1];
							domain->regions[region_index].data[offset++] = data[index];
						}
					}
				}
				region_index++;
			}
		}
	}
	assert(region_index == region_count);

	int64_t event_capacity = (128 + 4)*region_count;
	if (record_events) {
		events = malloc(sizeof *events + event_capacity*sizeof *events->data);
		assert(events != NULL); // NOTE: we can assert because event recording is disabled by default
		events->count = 0;
		events->capacity = event_capacity;
	}

	int64_t start = topologika_usec_counter();
	topologika_event_begin(events, topologika_event_color_gray, "Compute forest");

	// create scratch allocators
	//int64_t thread_count = processor_count();
	int64_t thread_count = omp_get_max_threads();
	stack_allocators = malloc(thread_count*sizeof *stack_allocators);
	if (stack_allocators == NULL) {
		goto out_of_memory;
	}
	int64_t const vertex_count = region_dims[0]*region_dims[1]*region_dims[2];
	// we inline data during sort so extra space for it
	// TODO(9/17/2019): how much memory we need for bedges? (in Vis 2019 paper it was *8 instead of *16, but
	//	it is not enough on small region resolutions such as 4x4x4)
	// TODO: more exact formula
	int64_t max_region_dim = topologika_max(region_dims[0], topologika_max(region_dims[1], region_dims[2]));
	size_t size = (2*vertex_count*sizeof(struct topologika_sort_vertex)) + sizeof (struct bedge)*max_region_dim*max_region_dim*16;
	{
		stack_allocators_memory = calloc(thread_count, size);
		if (stack_allocators_memory == NULL) {
			goto out_of_memory;
		}
	}
	for (int64_t i = 0; i < thread_count; i++) {
		stack_allocator_init(&stack_allocators[i], size, &stack_allocators_memory[i*size]);
	}

	forest = malloc(sizeof *forest + region_count*sizeof *forest->merge_trees);
	if (forest == NULL) {
		goto out_of_memory;
	}
	*forest = (struct topologika_merge_forest){
		.merge_tree_count = region_count,
	};

	assert(region_count < INT_MAX);
	int64_t i;
#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < region_count; i++) {
		struct region *region = &domain->regions[i];
		int tid = omp_get_thread_num();

		topologika_event_begin(events, topologika_event_color_green, "Compute MT");
		enum topologika_result result = compute_merge_tree(region, &stack_allocators[tid], events, region->dims, &forest->merge_trees[i]);
		// TODO(11/26/2019): how to bail from the parallel loop when computation fails
		assert(result == topologika_result_success);
		assert(stack_allocators[tid].offset == 0);
		topologika_event_end(events);

		topologika_event_begin(events, topologika_event_color_orange, "Compute RBS");
		result = compute_reduced_bridge_set(domain, i, &stack_allocators[tid], events, &forest->merge_trees[i].reduced_bridge_set);
		assert(result == topologika_result_success);
		assert(stack_allocators[tid].offset == 0);

		// build arc to bridge edges map
		{
			struct topologika_merge_tree *tree = &forest->merge_trees[i];
			struct reduced_bridge_set *set = tree->reduced_bridge_set;

			struct stack_allocation tmp_allocation = stack_allocator_alloc(&stack_allocators[tid], 8, set->edge_count*sizeof *set->edges);
			sort_edges(set->edge_count, set->edges, tmp_allocation.ptr, tree->vertex_to_arc);
			stack_allocator_free(&stack_allocators[tid], tmp_allocation);

			// TODO: merge into a single allocation? (more robust)
			tree->reduced_bridge_set_offsets = malloc(tree->arc_count*sizeof *tree->reduced_bridge_set_offsets);
			tree->reduced_bridge_set_counts = calloc(tree->arc_count, sizeof *tree->reduced_bridge_set_counts);
			assert(tree->reduced_bridge_set_offsets != NULL && tree->reduced_bridge_set_counts != NULL);
			for (int64_t edge_i = 0; edge_i < set->edge_count; edge_i++) {
				struct reduced_bridge_set_edge edge = set->edges[edge_i];
				topologika_local_t arc_id = tree->vertex_to_arc[edge.local_id];

				if (tree->reduced_bridge_set_counts[arc_id] == 0) {
					tree->reduced_bridge_set_offsets[arc_id] = edge_i;
				}
				tree->reduced_bridge_set_counts[arc_id]++;
			}
		}
		topologika_event_end(events);
	}
	free(stack_allocators);
	free(stack_allocators_memory);
	topologika_event_end(events);

	int64_t end = topologika_usec_counter();
	*out_construction_time = (end - start)*1e-6;

	if (record_events && events != NULL) {
		topologika_write_events("events.json", events);
		free(events);
	}

	*out_domain = domain;
	*out_forest = forest;
	return topologika_result_success;

out_of_memory:
	if (domain != NULL) {
		for (int64_t i = 0; i < region_count; i++) {
			free(domain->regions[i].data);
		}
	}
	free(domain);
	free(forest);
	free(events);
	free(stack_allocators_memory);
	free(stack_allocators);
	return topologika_error_out_of_memory;
}

void
topologika_domain_free(struct topologika_domain *domain)
{
	assert(domain != NULL);

	for (int64_t i = 0; i < domain->dims[0]*domain->dims[1]*domain->dims[2]; i++) {
		free(domain->regions[i].data);
	}
	free(domain);
}

void
topologika_merge_forest_free(struct topologika_merge_forest *forest)
{
	for (int64_t i = 0; i < forest->merge_tree_count; i++) {
		struct topologika_merge_tree *tree = &forest->merge_trees[i];
		free(tree->arcs);
		free(tree->segmentation_offsets);
		free(tree->segmentation_counts);
		free(tree->segmentation);
		free(tree->vertex_to_arc);
		free(tree->reduced_bridge_set);
		free(tree->reduced_bridge_set_offsets);
		free(tree->reduced_bridge_set_counts);
	}
	free(forest);
}

////////////////////// component max query //////////////////////////////////////

struct hash_set {
	int64_t count;
	int64_t capacity;
	uint64_t *keys;
};

void
hash_set_init(struct hash_set *table, uint64_t capacity)//, unsigned char *memory)
{
	assert(((capacity - 1) & capacity) == 0);
	table->count = 0;
	table->capacity = capacity;
	// TODO: pass in the memory or allocator
	table->keys = malloc(table->capacity*sizeof *table->keys);
	memset(table->keys, 0xFF, capacity*sizeof *table->keys);
}

void
hash_set_destroy(struct hash_set *table)
{
	free(table->keys);
}


// from murmur 3 (fmix64)
uint64_t
hash_set_hash(uint64_t key)
{
	key ^= key >> 33;
	key *= 0xFF51AFD7ED558CCDull;
	key ^= key >> 33;
	key *= 0xc4CEB9FE1A85EC53ull;
	key ^= key >> 33;
	return key;
}

void
hash_set_insert_no_check(struct hash_set *table, uint64_t key)
{
	uint64_t mask = table->capacity - 1;
	uint64_t hash = hash_set_hash(key);
	uint64_t idx = hash&mask;

	while (true) {
		if (table->keys[idx] == UINT64_MAX) {
			assert(table->count < table->capacity);
			table->keys[idx] = key;
			table->count++;
			return;
		} else if (table->keys[idx] == key) {
			return;
		}

		idx = (idx + 1)&mask;
	}
}


void
hash_set_grow(struct hash_set *table)
{
	uint64_t const old_capacity = table->capacity;
	uint64_t *old_keys = table->keys;

	table->count = 0;
	table->capacity *= 2;
	table->keys = malloc(table->capacity*sizeof *table->keys);
	memset(table->keys, 0xFF, table->capacity*sizeof *table->keys);

	for (uint64_t i = 0; i < old_capacity; i++) {
		if (old_keys[i] == UINT64_MAX) {
			continue;
		}
		hash_set_insert_no_check(table, old_keys[i]);
	}
	free(old_keys);
}


bool
hash_set_contains(struct hash_set const *table, uint64_t key)
{
	uint64_t mask = table->capacity - 1;
	uint64_t hash = hash_set_hash(key);
	uint64_t idx = hash&mask;

	while (true) {
		if (table->keys[idx] == UINT64_MAX) {
			return false;
		}
		if (table->keys[idx] == key) {
			return true;
		}

		idx = (idx + 1)&mask;
	}
}

void
hash_set_insert(struct hash_set *table, uint64_t key)
{
	if (2*table->count >= table->capacity) {
		hash_set_grow(table);
	}
	hash_set_insert_no_check(table, key);
}


struct set {
	struct hash_set data;
};

struct set *
set_create(void)
{
	struct set *set = malloc(sizeof *set);
	assert(set != NULL);
	hash_set_init(&set->data, 64);
	return set;
}

void
set_destroy(struct set *set)
{
	hash_set_destroy(&set->data);
	free(set);
}

bool
set_contains(struct set const *set, topologika_local_t vertex_id, uint32_t region_id)
{
	return hash_set_contains(&set->data, (((uint64_t)region_id) << 32) | ((uint64_t)vertex_id));
}


void
set_insert(struct set *set, topologika_local_t vertex_id, int64_t region_id)
{
	hash_set_insert(&set->data, (((uint64_t)region_id) << 32) | ((uint64_t)vertex_id));
}



// TODO: similar to the 'struct topologika_component'
struct worklist {
	int64_t count;
	int64_t capacity;
	struct worklist_item {
		topologika_local_t arc_id;
		topologika_local_t region_id;
	} items[];
};


// TODO: rename to topologika_valuevertex
struct component_max_result {
	topologika_data_t value;
	topologika_local_t vertex_index;
	topologika_local_t region_index;
};

bool
topologika_is_above(struct component_max_result v0, struct component_max_result v1)
{
	if (v0.value == v1.value) {
		return (v0.region_index == v1.region_index) ? v0.vertex_index > v1.vertex_index : v0.region_index > v1.region_index;
	}
	return v0.value > v1.value;
}

// TODO(2/27/2020): from_arc_id is not needed as it is always same as arc_id when the function is called
enum topologika_result
query_component_max_region_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo,
	struct component_max_result *out_max_vertex)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];
	struct component_max_result max_vertex = {.vertex_index = TOPOLOGIKA_LOCAL_MAX}; // BOTTOM

	// TODO(2/27/2020): we are overallocating potentially here, reducing the performance (should not be an issue
	//	if we use stack allocator)
	int64_t stack_capacity = tree->arc_count;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];

		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		struct component_max_result v = (struct component_max_result){.value = domain->regions[region_id].data[arc->max_vertex_id], .vertex_index = arc->max_vertex_id, .region_index = region_id,};
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(v, max_vertex)) {
			max_vertex = v;
		}

		// process neighbors
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] == item.from_arc_id) {
				continue;
			}
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){.arc_id = arc->children[i], .from_arc_id = item.arc_id};
		}
		// TODO(2/27/2020): simulation of simplicity will probably be needed when this query is used in
		//	the persistence query
		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id &&
			domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){.arc_id = arc->parent, .from_arc_id = item.arc_id};
		}

		// process reduced bridge set edges
		int64_t worklist_count = (*todo)->count;
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			if (domain->regions[region_id].data[edge.local_id] < threshold || domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id] < threshold) {
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					free(*todo);
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		if (worklist_count != (*todo)->count) {
			set_insert(visited, item.arc_id, region_id);
		}
	}
	free(stack);

	*out_max_vertex = max_vertex;

	return topologika_result_success;
}


// TODO: should we take pair vertex, threshold and do multiple queries?
enum topologika_result
topologika_query_componentmax(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_vertex *out_max_vertex)
{
	assert(domain != NULL && forest != NULL);

	// below threshold or thresholded merge tree
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < threshold) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success; // TODO(5/28/2020): return error?
	}

#if defined(TOPOLOGIKA_THRESHOLD)
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < TOPOLOGIKA_THRESHOLD) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success; // TODO(5/28/2020): return error?
	}
#endif

	// TODO(2/27/2020): scratch allocator (stack allocator assigned to a thread that executes the query)
	int64_t capacity = 1024;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	struct set *visited = set_create();

	struct component_max_result max_vertex = {0};

	// TODO(2/27/2020): abstract into traverse query
	// TODO(2/27/2020): push onto the worklist and handle inside the for loop below
	topologika_local_t arc_id = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index];
	enum topologika_result result = query_component_max_region_internal(domain, forest, threshold, arc_id, vertex.region_index, visited, &todo, &max_vertex);
	if (result != topologika_result_success) {
		free(todo);
		set_destroy(visited);
		return result;
	}

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		struct component_max_result tmp = {0};
		enum topologika_result result = query_component_max_region_internal(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, &tmp);
		if (result != topologika_result_success) {
			free(todo);
			set_destroy(visited);
			return result;
		}

		if (tmp.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			continue;
		}
		if (topologika_is_above(tmp, max_vertex)) {
			max_vertex = tmp;
		}
	}

	free(todo);
	set_destroy(visited);

	*out_max_vertex = (struct topologika_vertex){.vertex_index = max_vertex.vertex_index, .region_index = max_vertex.region_index};

	return topologika_result_success;
}




////////////////////// maxima query //////////////////////////////////////
// TODO: how to deal with memory alloc/dealloc for the query? (probably avoid malloc and free, take allocator as parameter?)
enum topologika_result
topologika_query_maxima(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex **out_maxima, int64_t *out_maximum_count)
{
	assert(domain != NULL && forest != NULL);

	int64_t maximum_count = 0;
	int64_t maximum_capacity = 1024;
	struct topologika_vertex *maxima = malloc(maximum_capacity*sizeof *maxima);
	if (maxima == NULL) {
		return topologika_error_out_of_memory;
	}

	for (int64_t region_index = 0; region_index < forest->merge_tree_count; region_index++) {
		struct topologika_merge_tree const *merge_tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < merge_tree->arc_count; arc_index++) {
			struct topologika_merge_tree_arc const *arc = &merge_tree->arcs[arc_index];
			if (arc->child_count != 0) {
				continue;
			}
#if defined(TOPOLOGIKA_DUMP)
			printf("region_id %d, arc_id %d\n", region_index, arc_index);
#endif
			struct component_max_result leaf = {
				.value = domain->regions[region_index].data[arc->max_vertex_id],
				.vertex_index = arc->max_vertex_id,
				.region_index = (topologika_local_t)region_index,
			};

			bool has_above_neighbor = false;
			for (int64_t i = 0; i < merge_tree->reduced_bridge_set_counts[arc_index]; i++) {
				struct reduced_bridge_set_edge edge = merge_tree->reduced_bridge_set->edges[merge_tree->reduced_bridge_set_offsets[arc_index] + i];
				assert(region_index != edge.neighbor_region_id); // NOTE: assumes convex regions?
#if defined(TOPOLOGIKA_DUMP)
				printf("\te%d %d\n", edge.local_id, edge.neighbor_local_id);
#endif
				// TODO: we could sort the bridge set edges per arc to break early (from
				//	earlier experiments it does not seem worth it; measure again)
				if (arc->max_vertex_id == edge.local_id) {
					struct component_max_result neighbor = {
						.value = domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id],
						.vertex_index = edge.neighbor_local_id,
						.region_index = edge.neighbor_region_id,
					};

					if (topologika_is_above(neighbor, leaf)) {
						has_above_neighbor = true;
						break;
					}
				}
			}
			if (has_above_neighbor) {
				continue;
			}
#if defined(TOPOLOGIKA_DUMP)
			printf("\tmaximum %d\n", arc->max_vertex_id);
#endif

			if (maximum_count == maximum_capacity) {
				maximum_capacity *= 2;
				struct topologika_vertex *tmp = realloc(maxima, maximum_capacity*sizeof *maxima);
				if (tmp == NULL) {
					free(maxima);
					return topologika_error_out_of_memory;
				}
				maxima = tmp;
			}
			maxima[maximum_count++] = (struct topologika_vertex){
				.vertex_index = arc->max_vertex_id,
				.region_index = (topologika_local_t)region_index,
			};
		}
	}

	*out_maxima = maxima;
	*out_maximum_count = maximum_count;
	return topologika_result_success;
}






////////////////////// components query //////////////////////////////////////
enum topologika_result
query_components_region_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t from_arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo,
	struct topologika_component **out_component)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];
	struct topologika_component *component = *out_component;

	// NOTE(11/25/2019): we do not use recursion to avoid stack overflow (the performance
	//	improvement is negligible for forest compared to a global tree)
	// TODO: stack allocator here, we should preallocate enough memory during forest
	//	construction that is sufficient for all temporary allocations during construction and queries
	int64_t stack_capacity = tree->arc_count;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = from_arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];

		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		// this part differs between the component max and component queries
		// callback(domain, forest, region_id, arc_id, context);

		bool crosses_threshold = !(arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold);

		// TODO(11/25/2019): we could store each segment as [region_id, count, local_id0, local_id1]
		// copy segmentation
		if (crosses_threshold) {
			// copy arc vertices above threshold
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];
				if (domain->regions[region_id].data[vertex_id] < threshold) {
					break;
				}

				if (component->count == component->capacity) {
					component->capacity *= 2;
					struct topologika_component *tmp = realloc(component, sizeof *component + component->capacity*sizeof *component->data);
					if (tmp == NULL) {
						free(stack);
						return topologika_error_out_of_memory;
					}
					component = tmp;
				}

				component->data[component->count++] = (struct topologika_vertex){
					.vertex_index = vertex_id,
					.region_index = region_id,
				};
			}
		} else {
			// copy whole arc
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];

				// TODO: resize to the nearest power of two > component->count + vertex_count outside of the loop
				if (component->count == component->capacity) {
					component->capacity *= 2;
					struct topologika_component *tmp = realloc(component, sizeof *component + component->capacity*sizeof *component->data);
					if (tmp == NULL) {
						free(stack);
						return topologika_error_out_of_memory;
					}
					component = tmp;
				}

				component->data[component->count++] = (struct topologika_vertex){
					.vertex_index = vertex_id,
					.region_index = region_id,
				};
			}
		}


		// push all neighbors and reduced bridge set edges if needed to worklist
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] ==  item.from_arc_id) {
				continue;
			}
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->children[i],
				.from_arc_id = item.arc_id,
			};
		}

		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id &&
			domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->parent,
				.from_arc_id = item.arc_id,
			};
		}

		int64_t worklist_count = (*todo)->count;
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			// TODO(11/25/2019): likely 2 cache misses
			if ((domain->regions[region_id].data[edge.local_id] < threshold) ||
				(domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id] < threshold)) {
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					// TODO: free
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// we enqueued some bridge set edges, so we need to mark the arc as visited to
		//	avoid potentially jumping back to it through some bridge set edge; also
		//	crossing threshold arcs need to be marked too to avoid starting the extraction
		//	of component from them (component in a forest can have multiple crossed arcs
		//	in contrast to the global tree where it is always only 1 crossed arc)
		if (worklist_count != (*todo)->count || crosses_threshold) {
			set_insert(visited, item.arc_id, region_id);
		}
	}
	free(stack);

	*out_component = component;

	return topologika_result_success;
}


enum topologika_result
query_components_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited,
	struct topologika_component **out_component)
{
	int64_t initial_component_capacity = 64;
	struct topologika_component *component = malloc(sizeof *component + initial_component_capacity*sizeof *component->data);
	if (component == NULL) {
		return topologika_error_out_of_memory;
	}
	component->count = 0;
	component->capacity = initial_component_capacity;

	// TODO: scratch alloc
	int64_t capacity = 1024;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		free(component);
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	enum topologika_result result = query_components_region_internal(domain, forest, threshold, arc_id, arc_id, region_id, visited, &todo, &component);
	if (result != topologika_result_success) {
		free(component);
		free(todo);
		return result;
	}

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		enum topologika_result result = query_components_region_internal(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, &component);
		if (result != topologika_result_success) {
			free(component);
			free(todo);
			return result;
		}
	}

	free(todo);

	*out_component = component;

	return topologika_result_success;
}


// TODO: the ***out_components is bit hairy (we need to return a pointer to array of pointers)
enum topologika_result
topologika_query_components(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_component ***out_components, int64_t *out_component_count)
{
	int64_t component_capacity = 16;
	struct topologika_component **components = malloc(component_capacity*sizeof *components);
	if (components == NULL) {
		return topologika_error_out_of_memory;
	}
	int64_t component_count = 0;

	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	for (int64_t region_index = 0; region_index < forest->merge_tree_count; region_index++) {
		struct topologika_merge_tree const *tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < tree->arc_count; arc_index++) {
			struct topologika_merge_tree_arc const *arc = &tree->arcs[arc_index];

			if (set_contains(visited, (topologika_local_t)arc_index, (topologika_local_t)region_index)) {
				continue;
			}

			// TODO: inline data values and profile
			if (domain->regions[region_index].data[arc->max_vertex_id] < threshold) {
				continue;
			}
			// the arc does not cross the threshold
			if (arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_index].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
				continue;
			}

			struct topologika_component *component = NULL;
			enum topologika_result result = query_components_internal(domain, forest, threshold, (topologika_local_t)arc_index, (topologika_local_t)region_index, visited, &component);
			if (result != topologika_result_success) {
				*out_components = 0;
				set_destroy(visited);
				return result;
			}

			// empty component
			if (component != NULL && component->count == 0) {
				free(component);
				continue;
			}

			if (component_count == component_capacity) {
				component_capacity *= 2;
				struct topologika_component **tmp = realloc(components, component_capacity*sizeof *tmp);
				if (tmp == NULL) {
					free(components);
					set_destroy(visited);
					return topologika_error_out_of_memory;
				}
				components = tmp;
			}
			components[component_count++] = component;
		}
	}

	set_destroy(visited);

	*out_components = components;
	*out_component_count = component_count;

	return topologika_result_success;
}




////////////////////// components ids query //////////////////////////////////////
enum topologika_result
query_components_ids_region_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t from_arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo,
	struct topologika_vertex *out_component)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];

	// NOTE(11/25/2019): we do not use recursion to avoid stack overflow (the performance
	//	improvement is negligible for forest compared to a global tree)
	// TODO: stack allocator here, we should preallocate enough memory during forest
	//	construction that is sufficient for all temporary allocations during construction and queries
	int64_t stack_capacity = tree->arc_count;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = from_arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];

		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		// TODO: we could return the highest vertex here
		if (out_component->region_index == TOPOLOGIKA_LOCAL_MAX) {
			*out_component = (struct topologika_vertex){.region_index = region_id, .vertex_index = arc->max_vertex_id};
		}

		// push all neighbors and reduced bridge set edges if needed to worklist
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] ==  item.from_arc_id) {
				continue;
			}
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->children[i],
				.from_arc_id = item.arc_id,
			};
		}

		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id &&
			domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->parent,
				.from_arc_id = item.arc_id,
			};
		}

		int64_t worklist_count = (*todo)->count;
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			// TODO(11/25/2019): likely 2 cache misses
			if ((domain->regions[region_id].data[edge.local_id] < threshold) ||
				(domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id] < threshold)) {
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					// TODO: free
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// we enqueued some bridge set edges, so we need to mark the arc as visited to
		//	avoid potentially jumping back to it through some bridge set edge; also
		//	crossing threshold arcs need to be marked too to avoid starting the extraction
		//	of component from them (component in a forest can have multiple crossed arcs
		//	in contrast to the global tree where it is always only 1 crossed arc)
		bool crosses_threshold = !(arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold);
		if (worklist_count != (*todo)->count || crosses_threshold) {
			set_insert(visited, item.arc_id, region_id);
		}
	}
	free(stack);

	return topologika_result_success;
}


enum topologika_result
query_components_ids_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited,
	struct topologika_vertex *out_component)
{
	// TODO: scratch alloc
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	enum topologika_result result = query_components_ids_region_internal(domain, forest, threshold, arc_id, arc_id, region_id, visited, &todo, out_component);
	if (result != topologika_result_success) {
		free(todo);
		return result;
	}

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		enum topologika_result result = query_components_ids_region_internal(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, out_component);
		if (result != topologika_result_success) {
			free(todo);
			return result;
		}
	}

	free(todo);

	return topologika_result_success;
}


enum topologika_result
topologika_query_components_ids(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_vertex **out_components, int64_t *out_component_count)
{
	int64_t component_capacity = 16;
	struct topologika_vertex *components = malloc(component_capacity*sizeof *components);
	if (components == NULL) {
		return topologika_error_out_of_memory;
	}
	int64_t component_count = 0;

	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	for (int64_t region_index = 0; region_index < forest->merge_tree_count; region_index++) {
		struct topologika_merge_tree const *tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < tree->arc_count; arc_index++) {
			struct topologika_merge_tree_arc const *arc = &tree->arcs[arc_index];

			if (set_contains(visited, (topologika_local_t)arc_index, (topologika_local_t)region_index)) {
				continue;
			}

			// TODO: inline data values and profile
			if (domain->regions[region_index].data[arc->max_vertex_id] < threshold) {
				continue;
			}
			// the arc does not cross the threshold
			if (arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_index].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
				continue;
			}

			struct topologika_vertex component_id = topologika_bottom;
			enum topologika_result result = query_components_ids_internal(domain, forest, threshold, (topologika_local_t)arc_index, (topologika_local_t)region_index, visited, &component_id);
			if (result != topologika_result_success) {
				*out_components = 0;
				set_destroy(visited);
				return result;
			}

			// empty component
			if (component_id.region_index == TOPOLOGIKA_LOCAL_MAX && component_id.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
				continue;
			}

			if (component_count == component_capacity) {
				component_capacity *= 2;
				struct topologika_vertex *tmp = realloc(components, component_capacity*sizeof *tmp);
				if (tmp == NULL) {
					free(components);
					set_destroy(visited);
					return topologika_error_out_of_memory;
				}
				components = tmp;
			}
			components[component_count++] = component_id;
		}
	}

	set_destroy(visited);

	*out_components = components;
	*out_component_count = component_count;

	return topologika_result_success;
}




////////////////////////// component query /////////////////////////
enum topologika_result
topologika_query_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_component **out_component)
{
	assert(vertex.region_index >= 0 && vertex.region_index < forest->merge_tree_count);
	struct topologika_merge_tree const *tree = &forest->merge_trees[vertex.region_index];

	// TODO(11/26/2019): assert that vertex_index is valid
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < threshold) {
		return topologika_error_no_output;
	}

	topologika_local_t arc_index = tree->vertex_to_arc[vertex.vertex_index];

	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	struct topologika_component *component = NULL;
	enum topologika_result result = query_components_internal(domain, forest, threshold, arc_index, vertex.region_index, visited, &component);

	set_destroy(visited);

	if (result != topologika_result_success) {
		return result;
	}
	if (component->count == 0) {
		free(component);
		return topologika_error_no_output;
	}

	*out_component = component;
	return topologika_result_success;
}


///////////////////////// traverse component query ////////////////////
enum topologika_result
traverse_component_region_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,  topologika_local_t arc_id, topologika_local_t from_arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc))
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];

	// NOTE(11/25/2019): we do not use recursion to avoid stack overflow (the performance
	//	improvement is negligible for forest compared to a global tree)
	// TODO: stack allocator here, we should preallocate enough memory during forest
	//	construction that is sufficient for all temporary allocations during construction and queries
	int64_t stack_capacity = tree->arc_count;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = from_arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];

		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		function(context, arc);

		// this part differs between the component max and component queries
		// callback(domain, forest, region_id, arc_id, context);

		bool crosses_threshold = !(arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold);

		// TODO(11/25/2019): we could store each segment as [region_id, count, local_id0, local_id1]
		// copy segmentation
		if (crosses_threshold) {
			// copy arc vertices above threshold
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];
				if (domain->regions[region_id].data[vertex_id] < threshold) {
					break;
				}
			}
		} else {
			// copy whole arc
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];
			}
		}


		// push all neighbors and reduced bridge set edges if needed to worklist
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] ==  item.from_arc_id) {
				continue;
			}
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->children[i],
				.from_arc_id = item.arc_id,
			};
		}

		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id &&
			domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->parent,
				.from_arc_id = item.arc_id,
			};
		}

		int64_t worklist_count = (*todo)->count;
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			// TODO(11/25/2019): likely 2 cache misses
			if ((domain->regions[region_id].data[edge.local_id] < threshold) ||
				(domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id] < threshold)) {
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					// TODO: free
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// we enqueued some bridge set edges, so we need to mark the arc as visited to
		//	avoid potentially jumping back to it through some bridge set edge; also
		//	crossing threshold arcs need to be marked too to avoid starting the extraction
		//	of component from them (component in a forest can have multiple crossed arcs
		//	in contrast to the global tree where it is always only 1 crossed arc)
		if (worklist_count != (*todo)->count || crosses_threshold) {
			set_insert(visited, item.arc_id, region_id);
		}
	}
	free(stack);

	return topologika_result_success;
}

enum topologika_result
traverse_components_internal(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc))
{
	// TODO: scratch alloc
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	enum topologika_result result = traverse_component_region_internal(domain, forest, threshold, arc_id, arc_id, region_id, visited, &todo, context, function);
	if (result != topologika_result_success) {
		free(todo);
		return result;
	}

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		enum topologika_result result = traverse_component_region_internal(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, context, function);
		if (result != topologika_result_success) {
			free(todo);
			return result;
		}
	}

	free(todo);

	return topologika_result_success;
}


enum topologika_result
topologika_traverse_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc))
{
	assert(vertex.region_index >= 0 && vertex.region_index < forest->merge_tree_count);
	struct topologika_merge_tree const *tree = &forest->merge_trees[vertex.region_index];

	// TODO(11/26/2019): assert that vertex_index is valid
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < threshold) {
		return topologika_error_no_output;
	}

	topologika_local_t arc_index = tree->vertex_to_arc[vertex.vertex_index];

	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	enum topologika_result result = traverse_components_internal(domain, forest, threshold, arc_index, vertex.region_index, visited, context, function);

	set_destroy(visited);

	return result;
}


enum topologika_result
topologika_traverse_components(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, void *context, void (*function)(void *context, struct topologika_merge_tree_arc const *arc))
{
	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	for (int64_t region_index = 0; region_index < forest->merge_tree_count; region_index++) {
		struct topologika_merge_tree const *tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < tree->arc_count; arc_index++) {
			struct topologika_merge_tree_arc const *arc = &tree->arcs[arc_index];

			if (set_contains(visited, (topologika_local_t)arc_index, (topologika_local_t)region_index)) {
				continue;
			}

			// TODO: inline data values and profile
			if (domain->regions[region_index].data[arc->max_vertex_id] < threshold) {
				continue;
			}
			// the arc does not cross the threshold
			if (arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_index].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
				continue;
			}

			enum topologika_result result = traverse_components_internal(domain, forest, threshold, (topologika_local_t)arc_index, (topologika_local_t)region_index, visited, context, function);
			if (result != topologika_result_success) {
				set_destroy(visited);
				return result;
			}
		}
	}

	set_destroy(visited);

	return topologika_result_success;
}





////////////////////////// triplet query //////////////////////////////
struct topologika_item {
	topologika_data_t value;
	topologika_local_t region_index, vertex_index;
	topologika_local_t arc_index, arc_region;
};

bool
is_above_(struct topologika_item a, struct topologika_item b)
{
	return topologika_is_above((struct component_max_result){.value = a.value, .region_index = a.region_index, .vertex_index = a.vertex_index}, (struct component_max_result){.value = b.value, .region_index = b.region_index, .vertex_index = b.vertex_index});
}

struct topologika_priority_queue {
	struct topologika_item *array;
	int64_t count, capacity;
};


void
topologika_priority_queue_init(struct topologika_priority_queue *pq)
{
	pq->count = 1; // NOTE(3/10/2020): index starting from 1 to simplify parent index computation; TODO: is it worth it?
	pq->capacity = 8;
	pq->array = malloc(pq->capacity*sizeof *pq->array);
	assert(pq->array != NULL);
}

bool
topologika_priority_queue_is_empty(struct topologika_priority_queue const *pq)
{
	assert(pq->count >= 1);
	return pq->count == 1;
}

// TODO(3/10/2020): pass in allocator
void
topologika_priority_queue_enqueue(struct topologika_priority_queue *pq, struct topologika_item item)
{
	if (pq->count == pq->capacity) {
		pq->capacity *= 2;
		struct topologika_item *tmp = realloc(pq->array, pq->capacity*sizeof *pq->array);
		assert(tmp != NULL);
		pq->array = tmp;
	}

// TODO(3/10/2020): is it worth the macro to improve the readability of code below?
#define PARENT(index) (index/2)

	int64_t index = pq->count++;
	while (index > 1 && is_above_(item, pq->array[PARENT(index)])) {
		pq->array[index] = pq->array[PARENT(index)];
		index = PARENT(index);
	}
	pq->array[index] = item;

#undef PARENT
}

struct topologika_item
topologika_priority_queue_dequeue(struct topologika_priority_queue *pq)
{
	assert(pq->count >= 1);

	struct topologika_item max = pq->array[1];
	pq->array[1] = pq->array[--pq->count];

	// heapify
#define LEFT(index) (index*2)
#define RIGHT(index) (index*2 + 1)

	int64_t index = 1;
	int64_t largest = index;
	while (true) {
		if (LEFT(index) < pq->count && is_above_(pq->array[LEFT(index)], pq->array[largest])) {
			largest = LEFT(index);
		}
		if (RIGHT(index) < pq->count && is_above_(pq->array[RIGHT(index)], pq->array[largest])) {
			largest = RIGHT(index);
		} 
		if (largest == index) {
			break;
		}

		// swap
		{
			struct topologika_item tmp = pq->array[index];
			pq->array[index] = pq->array[largest];
			pq->array[largest] = tmp;
		}
		index = largest;
	}

#undef LEFT
#undef RIGHT

	return max;
}


enum topologika_result
query_component_max_region_internal_(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct component_max_result threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo, struct topologika_priority_queue *pq,
	struct component_max_result *out_max_vertex)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];
	struct component_max_result max_vertex = {.vertex_index = TOPOLOGIKA_LOCAL_MAX}; // BOTTOM

	// TODO(3/16/2021): we should be using stack allocator instead of malloc+realloc
	int64_t stack_capacity = 1024;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];
		if (set_contains(visited, item.arc_id, region_id)) {
			continue;
		}

		//printf("\titem arc (%d, %d) from arc (%d, %d)\n", region_id, item.arc_id, region_id, item.from_arc_id);
		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		struct component_max_result v = (struct component_max_result){.value = domain->regions[region_id].data[arc->max_vertex_id], .vertex_index = arc->max_vertex_id, .region_index = region_id,};
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(v, max_vertex)) {
			max_vertex = v;
		}

		// process neighbors
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] == item.from_arc_id) {
				continue;
			}
			if (stack_count == stack_capacity) {
				stack_capacity *= 2;
				stack = realloc(stack, sizeof *stack*stack_capacity);
				assert(stack != NULL);
			}
			stack[stack_count++] = (struct stack_item){.arc_id = arc->children[i], .from_arc_id = item.arc_id};
		}
		// TODO(2/27/2020): simulation of simplicity will probably be needed when this query is used in
		//	the persistence query
		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id) {
			struct component_max_result parent = {
				.value = domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id],
				.region_index = region_id,
				.vertex_index = tree->arcs[arc->parent].max_vertex_id,
			};
			// parent >= threshold => !(parent < threshold) => !(threshold > parent)
			if (!topologika_is_above(threshold, parent)) {
				if (stack_count == stack_capacity) {
					stack_capacity *= 2;
					stack = realloc(stack, sizeof *stack*stack_capacity);
					assert(stack != NULL);
				}
				stack[stack_count++] = (struct stack_item){.arc_id = arc->parent, .from_arc_id = item.arc_id};
			} else {
				topologika_priority_queue_enqueue(pq, (struct topologika_item) {
					.value = parent.value,
					.region_index = parent.region_index,
					.vertex_index = parent.vertex_index,
					.arc_index = arc->parent,
					.arc_region = region_id,
				});
			}
		}

		// process reduced bridge set edges
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			struct component_max_result local = {
				.value = domain->regions[region_id].data[edge.local_id],
				.region_index = region_id,
				.vertex_index = edge.local_id,
			};
			struct component_max_result neighbor = {
				.value = domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id],
				.region_index = edge.neighbor_region_id,
				.vertex_index = edge.neighbor_local_id,
			};

			// local or neighbor is below threshold => does not get visited current query and needs to be enqueued for later processing
			if (topologika_is_above(threshold, local) || topologika_is_above(threshold, neighbor)) {
				//printf("\tlocal (%d, %d), neighbor (%d, %d)\n", local.region_index, local.vertex_index, neighbor.region_index, neighbor.vertex_index);
				if (topologika_is_above(neighbor, local)) {
					topologika_priority_queue_enqueue(pq, (struct topologika_item){
						.value = domain->regions[region_id].data[edge.local_id],
						.region_index = region_id,
						.vertex_index = edge.local_id,
						.arc_index = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
						.arc_region = edge.neighbor_region_id,
					});
				} else {
					topologika_priority_queue_enqueue(pq, (struct topologika_item){
						.value = domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id],
						.region_index = edge.neighbor_region_id,
						.vertex_index = edge.neighbor_local_id,
						.arc_index = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
						.arc_region = edge.neighbor_region_id,
					});
				}
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					free(*todo);
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// insert every visited arc to avoid repeated traversal => O(n^2) during triplet query
		/*
		printf("\tInserting arc (%d, %d) to visited set:\n", region_id, item.arc_id);
		printf("\t\tchild count %d\n", arc->child_count);
		printf("\t\tparent (%d, %d) hv (%d, %d)\n", region_id, arc->parent, region_id, arc->max_vertex_id);
		for (int64_t i = tree->reduced_bridge_set_offsets[item.arc_id]; i < tree->reduced_bridge_set_offsets[item.arc_id] + tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge e = tree->reduced_bridge_set->edges[i];
			printf("\t\tedge local %f (%d, %d), neighbor %f (%d, %d)\n", domain->regions[region_id].data[e.local_id], region_id, e.local_id, domain->regions[e.neighbor_region_id].data[e.neighbor_local_id], e.neighbor_region_id, e.neighbor_local_id);
		}
		*/
		set_insert(visited, item.arc_id, region_id);
	}
	free(stack);

	*out_max_vertex = max_vertex;

	return topologika_result_success;
}


// TODO: should we take pair vertex, threshold and do multiple queries?
enum topologika_result
topologika_query_component_max_(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, topologika_local_t arc_region, topologika_local_t arc_index, struct component_max_result threshold, struct set *visited, struct topologika_priority_queue *pq, struct component_max_result max_vertex,
	struct topologika_vertex *out_max_vertex)
{
	assert(domain != NULL && forest != NULL);

	//printf("COMPMAX: threshold %f (%d, %d), arc (%d, %d)\n", threshold.value, threshold.region_index, threshold.vertex_index, arc_region, arc_index);

#if defined(TOPOLOGIKA_THRESHOLD)
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < TOPOLOGIKA_THRESHOLD) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success; // TODO(5/28/2020): return error?
	}
#endif

	// TODO(2/27/2020): scratch allocator (stack allocator assigned to a thread that executes the query)
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	// TODO(2/27/2020): abstract into traverse query
	// TODO: SoS for the assert
	assert(domain->regions[arc_region].data[forest->merge_trees[arc_region].arcs[arc_index].max_vertex_id] >= threshold.value);
	todo->items[todo->count++] = (struct worklist_item){
		.arc_id = arc_index,
		.region_id = arc_region,
	};

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		struct component_max_result tmp = {0};
		enum topologika_result result = query_component_max_region_internal_(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, &tmp);
		if (result != topologika_result_success) {
			free(todo);
			return result;
		}

		if (tmp.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			continue;
		}
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(tmp, max_vertex)) {
			max_vertex = tmp;
		}
	}

	free(todo);

	*out_max_vertex = (struct topologika_vertex){.vertex_index = max_vertex.vertex_index, .region_index = max_vertex.region_index};

	return topologika_result_success;
}


// TODO(6/25/2020): bridge set edges are pushed onto priority queue twice (because they are duplicated for each local tree)
enum topologika_result
topologika_query_triplet(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	struct topologika_triplet *out_triplet)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);
	
	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
		.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
		.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();

	struct component_max_result current_max = {.vertex_index = TOPOLOGIKA_LOCAL_MAX};
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		// we pass in current_max because there can be a merge saddle incident to reduced bridge set edge, and will execute in
		//	multiple component max queries, and we need the highest max of those
		struct topologika_vertex max = {0};
		topologika_query_component_max_(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			// NOTE(6/22/2020): we need to get the highest max for query if there are multiple reduced bridge set edges with the same lower end vertex
			if (!topologika_priority_queue_is_empty(&pq) && pq.array[1].region_index == item.region_index && pq.array[1].vertex_index == item.vertex_index) {
				continue;
			}
			if (topologika_is_above(m, v)) {
				*out_triplet = (struct topologika_triplet){
					.u = vertex,
					.s = {.region_index = item.region_index, .vertex_index = item.vertex_index},
					.v = max,
				};
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_triplet = (struct topologika_triplet){
		.u = vertex,
		.s = vertex,
		.v = vertex,
	};

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}


///////////////// triplet query with runtime statistics (visited regions,...) //////////////
struct topologika_query_statistics {
	struct hash_set visited_regions;
	int64_t priority_queue_max_count;
	int64_t visited_set_max_count;
	int64_t componentmax_call_count;
	int64_t todo_max_count;
};

// TODO: should we take pair vertex, threshold and do multiple queries?
// TODO(2/27/2020): should no result be an error or success (bottom)
enum topologika_result
topologika_query_component_max_with_statistics(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, topologika_local_t arc_region, topologika_local_t arc_index, struct component_max_result threshold, struct set *visited, struct topologika_priority_queue *pq, struct component_max_result max_vertex,
	struct topologika_vertex *out_max_vertex, struct topologika_query_statistics *out_statistics)
{
	assert(domain != NULL && forest != NULL);

	//printf("COMPMAX: threshold %f (%d, %d), arc (%d, %d)\n", threshold.value, threshold.region_index, threshold.vertex_index, arc_region, arc_index);

#if defined(TOPOLOGIKA_THRESHOLD)
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < TOPOLOGIKA_THRESHOLD) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success; // TODO(5/28/2020): return error?
	}
#endif

	// TODO(2/27/2020): scratch allocator (stack allocator assigned to a thread that executes the query)
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	int64_t todo_max_count = 0;

	// TODO(2/27/2020): abstract into traverse query
	// TODO: SoS for the assert
	assert(domain->regions[arc_region].data[forest->merge_trees[arc_region].arcs[arc_index].max_vertex_id] >= threshold.value);
	todo->items[todo->count++] = (struct worklist_item){
		.arc_id = arc_index,
		.region_id = arc_region,
	};

	for (int64_t i = 0; i < todo->count; i++) {
		if (todo->count > todo_max_count) {
			todo_max_count = todo->count;
		}

		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		hash_set_insert(&out_statistics->visited_regions, todo->items[i].region_id);

		struct component_max_result tmp = {0};
		enum topologika_result result = query_component_max_region_internal_(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, &tmp);
		if (result != topologika_result_success) {
			out_statistics->todo_max_count = todo_max_count;
			free(todo);
			return result;
		}

		if (tmp.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			continue;
		}
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(tmp, max_vertex)) {
			max_vertex = tmp;
		}
	}

	free(todo);

	out_statistics->todo_max_count = todo_max_count;

	*out_max_vertex = (struct topologika_vertex){.vertex_index = max_vertex.vertex_index, .region_index = max_vertex.region_index};

	return topologika_result_success;
}

// TODO(6/25/2020): bridge set edges are pushed onto priority queue twice (because they are duplicated for each local tree)
enum topologika_result
topologika_query_triplet_with_statistics(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	struct topologika_triplet *out_triplet, struct topologika_query_statistics *out_statistics)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);
	
	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
		.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
		.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();
	// TODO(3/1/2021): we could extract visited regions directly from the visited set, but that requires
	//	knowledge of the internals of the set data structure; on the other hand, getting the correct set
	//	is less error prone
	hash_set_init(&out_statistics->visited_regions, 64);

	out_statistics->priority_queue_max_count = pq.count - 1; // empty priority queue has count == 1
	out_statistics->visited_set_max_count = visited->data.count;
	out_statistics->componentmax_call_count = 0;
	out_statistics->todo_max_count = 0;

	struct component_max_result current_max = {.vertex_index = TOPOLOGIKA_LOCAL_MAX};
	while (!topologika_priority_queue_is_empty(&pq)) {
		if (out_statistics->priority_queue_max_count < pq.count - 1) {
			out_statistics->priority_queue_max_count = pq.count - 1;
		}
		if (out_statistics->visited_set_max_count < visited->data.count) {
			out_statistics->visited_set_max_count = visited->data.count;
		}

		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		struct topologika_vertex max = {0};
		topologika_query_component_max_with_statistics(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max, out_statistics);
		out_statistics->componentmax_call_count++;
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			// NOTE(6/22/2020): we need to get the highest max for query if there are multiple reduced bridge set edges with the same lower end vertex
			if (!topologika_priority_queue_is_empty(&pq) && pq.array[1].region_index == item.region_index && pq.array[1].vertex_index == item.vertex_index) {
				continue;
			}
			if (topologika_is_above(m, v)) {
				*out_triplet = (struct topologika_triplet){
					.u = vertex,
					.s = {.region_index = item.region_index, .vertex_index = item.vertex_index},
					.v = max,
				};
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_triplet = (struct topologika_triplet){
		.u = vertex,
		.s = vertex,
		.v = vertex,
	};

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}




/////////////////////////////// persistence query //////////////////////////////////
// TODO(6/25/2020): somehow merge with the triplet query (the only difference is that in persistence query we can terminate after
//	finding any vertex above the queries vertex, thus we can avoid traversing to the other above vertices)
// TODO: terminate early when we find vertex with value greater than the input vertex
enum topologika_result
topologika_query_persistence_without_early_termination(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	double *out_persistence)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
			.region_index = vertex.region_index,
			.vertex_index = vertex.vertex_index,
			.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
			.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		struct topologika_vertex max = {0};
		topologika_query_component_max_(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				*out_persistence = v.value - item.value;
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistence = INFINITY;

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}


////////////////////// persistence query with early termination when higher vertex is found /////////////////
enum topologika_result
query_component_max_region_internal_terminate_early(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct component_max_result threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo, struct topologika_priority_queue *pq,
	struct component_max_result maximum_vertex, struct component_max_result *out_max_vertex)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];
	struct component_max_result max_vertex = {.vertex_index = TOPOLOGIKA_LOCAL_MAX}; // BOTTOM

	// TODO(3/16/2021): we should be using stack allocator instead of malloc+realloc
	int64_t stack_capacity = 1024;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];
		if (set_contains(visited, item.arc_id, region_id)) {
			continue;
		}

		//printf("\titem arc (%d, %d) from arc (%d, %d)\n", region_id, item.arc_id, region_id, item.from_arc_id);
		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		struct component_max_result v = (struct component_max_result){.value = domain->regions[region_id].data[arc->max_vertex_id], .vertex_index = arc->max_vertex_id, .region_index = region_id,};
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(v, max_vertex)) {
			max_vertex = v;
		}
		if (topologika_is_above(max_vertex, maximum_vertex)) {
			break;
		}

		// process neighbors
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] == item.from_arc_id) {
				continue;
			}
			if (stack_count == stack_capacity) {
				stack_capacity *= 2;
				stack = realloc(stack, sizeof *stack*stack_capacity);
				assert(stack != NULL);
			}
			stack[stack_count++] = (struct stack_item){.arc_id = arc->children[i], .from_arc_id = item.arc_id};
		}
		// TODO(2/27/2020): simulation of simplicity will probably be needed when this query is used in
		//	the persistence query
		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id) {
			struct component_max_result parent = {
				.value = domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id],
				.region_index = region_id,
				.vertex_index = tree->arcs[arc->parent].max_vertex_id,
			};
			// parent >= threshold => !(parent < threshold) => !(threshold > parent)
			if (!topologika_is_above(threshold, parent)) {
				if (stack_count == stack_capacity) {
					stack_capacity *= 2;
					stack = realloc(stack, sizeof *stack*stack_capacity);
					assert(stack != NULL);
				}
				stack[stack_count++] = (struct stack_item){.arc_id = arc->parent, .from_arc_id = item.arc_id};
			} else {
				topologika_priority_queue_enqueue(pq, (struct topologika_item) {
					.value = parent.value,
					.region_index = parent.region_index,
					.vertex_index = parent.vertex_index,
					.arc_index = arc->parent,
					.arc_region = region_id,
				});
			}
		}

		// process reduced bridge set edges
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			struct component_max_result local = {
				.value = domain->regions[region_id].data[edge.local_id],
				.region_index = region_id,
				.vertex_index = edge.local_id,
			};
			struct component_max_result neighbor = {
				.value = domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id],
				.region_index = edge.neighbor_region_id,
				.vertex_index = edge.neighbor_local_id,
			};

			// local or neighbor is below threshold => does not get visited current query and needs to be enqueued for later processing
			if (topologika_is_above(threshold, local) || topologika_is_above(threshold, neighbor)) {
				//printf("\tlocal (%d, %d), neighbor (%d, %d)\n", local.region_index, local.vertex_index, neighbor.region_index, neighbor.vertex_index);
				if (topologika_is_above(neighbor, local)) {
					topologika_priority_queue_enqueue(pq, (struct topologika_item){
						.value = domain->regions[region_id].data[edge.local_id],
						.region_index = region_id,
						.vertex_index = edge.local_id,
						.arc_index = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
						.arc_region = edge.neighbor_region_id,
					});
				} else {
					topologika_priority_queue_enqueue(pq, (struct topologika_item){
						.value = domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id],
						.region_index = edge.neighbor_region_id,
						.vertex_index = edge.neighbor_local_id,
						.arc_index = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
						.arc_region = edge.neighbor_region_id,
					});
				}
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					free(*todo);
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// insert every visited arc to avoid repeated traversal => O(n^2) during triplet query
		/*
		printf("\tInserting arc (%d, %d) to visited set:\n", region_id, item.arc_id);
		printf("\t\tchild count %d\n", arc->child_count);
		printf("\t\tparent (%d, %d) hv (%d, %d)\n", region_id, arc->parent, region_id, arc->max_vertex_id);
		for (int64_t i = tree->reduced_bridge_set_offsets[item.arc_id]; i < tree->reduced_bridge_set_offsets[item.arc_id] + tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge e = tree->reduced_bridge_set->edges[i];
			printf("\t\tedge local %f (%d, %d), neighbor %f (%d, %d)\n", domain->regions[region_id].data[e.local_id], region_id, e.local_id, domain->regions[e.neighbor_region_id].data[e.neighbor_local_id], e.neighbor_region_id, e.neighbor_local_id);
		}
		*/
		set_insert(visited, item.arc_id, region_id);
	}
	free(stack);

	*out_max_vertex = max_vertex;

	return topologika_result_success;
}


enum topologika_result
topologika_query_component_max_terminate_early(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, topologika_local_t arc_region, topologika_local_t arc_index, struct component_max_result threshold, struct set *visited, struct topologika_priority_queue *pq, struct component_max_result max_vertex,
	struct topologika_vertex *out_max_vertex)
{
	assert(domain != NULL && forest != NULL);

	//printf("COMPMAX: threshold %f (%d, %d), arc (%d, %d)\n", threshold.value, threshold.region_index, threshold.vertex_index, arc_region, arc_index);

#if defined(TOPOLOGIKA_THRESHOLD)
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < TOPOLOGIKA_THRESHOLD) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success;
	}
#endif

	// TODO(2/27/2020): scratch allocator (stack allocator assigned to a thread that executes the query)
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	// TODO(2/27/2020): abstract into traverse query
	// TODO: SoS for the assert
	assert(domain->regions[arc_region].data[forest->merge_trees[arc_region].arcs[arc_index].max_vertex_id] >= threshold.value);
	todo->items[todo->count++] = (struct worklist_item){
		.arc_id = arc_index,
		.region_id = arc_region,
	};

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		struct component_max_result tmp = {0};
		//enum topologika_result result = query_component_max_region_internal_(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, &tmp);
		enum topologika_result result = query_component_max_region_internal_terminate_early(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, max_vertex, &tmp);
		if (result != topologika_result_success) {
			free(todo);
			return result;
		}

		if (tmp.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			continue;
		}
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(tmp, max_vertex)) {
			max_vertex = tmp;
			break;
		}
	}

	free(todo);

	*out_max_vertex = (struct topologika_vertex){.vertex_index = max_vertex.vertex_index, .region_index = max_vertex.region_index};

	return topologika_result_success;
}


// TODO(3/27/2021): could be implemented via the persistencebelow query where the simplification threshold is INFINITY,
//	maybe do not even expose this query via the API?
// TODO(6/25/2020): somehow merge with the triplet query (the only difference is that in persistence query we can terminate after
//	finding any vertex above the queries vertex, thus we can avoid traversing to the other above vertices)
// TODO: terminate early when we find vertex with value greater than the input vertex
enum topologika_result
topologika_query_persistence(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	double *out_persistence)//, struct topologika_vertex *u, struct topologika_vertex *s)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
			.region_index = vertex.region_index,
			.vertex_index = vertex.vertex_index,
			.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
			.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		struct topologika_vertex max = {0};
		topologika_query_component_max_terminate_early(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				//*u = (struct topologika_vertex){.region_index = v.region_index, .vertex_index = v.vertex_index};
				//*s = (struct topologika_vertex){.region_index = item.region_index, .vertex_index = item.vertex_index};
				*out_persistence = v.value - item.value;
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistence = INFINITY;

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}


// with statistics
enum topologika_result
topologika_query_component_max_terminate_early_with_statistics(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, topologika_local_t arc_region, topologika_local_t arc_index, struct component_max_result threshold, struct set *visited, struct topologika_priority_queue *pq, struct component_max_result max_vertex,
	struct topologika_vertex *out_max_vertex, struct topologika_query_statistics *out_statistics)
{
	assert(domain != NULL && forest != NULL);

	//printf("COMPMAX: threshold %f (%d, %d), arc (%d, %d)\n", threshold.value, threshold.region_index, threshold.vertex_index, arc_region, arc_index);

#if defined(TOPOLOGIKA_THRESHOLD)
	if (domain->regions[vertex.region_index].data[vertex.vertex_index] < TOPOLOGIKA_THRESHOLD) {
		*out_max_vertex = topologika_bottom;
		return topologika_result_success;
	}
#endif

	// TODO(2/27/2020): scratch allocator (stack allocator assigned to a thread that executes the query)
	int64_t capacity = 64;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	// TODO(2/27/2020): abstract into traverse query
	// TODO: SoS for the assert
	assert(domain->regions[arc_region].data[forest->merge_trees[arc_region].arcs[arc_index].max_vertex_id] >= threshold.value);
	todo->items[todo->count++] = (struct worklist_item){
		.arc_id = arc_index,
		.region_id = arc_region,
	};

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		hash_set_insert(&out_statistics->visited_regions, todo->items[i].region_id);

		struct component_max_result tmp = {0};
		//enum topologika_result result = query_component_max_region_internal_(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, &tmp);
		enum topologika_result result = query_component_max_region_internal_terminate_early(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, pq, max_vertex, &tmp);
		if (result != topologika_result_success) {
			free(todo);
			return result;
		}

		if (tmp.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			continue;
		}
		if (max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX || topologika_is_above(tmp, max_vertex)) {
			max_vertex = tmp;
			break;
		}
	}

	free(todo);

	*out_max_vertex = (struct topologika_vertex){.vertex_index = max_vertex.vertex_index, .region_index = max_vertex.region_index};

	return topologika_result_success;
}


// TODO(6/25/2020): somehow merge with the triplet query (the only difference is that in persistence query we can terminate after
//	finding any vertex above the queries vertex, thus we can avoid traversing to the other above vertices)
// TODO: terminate early when we find vertex with value greater than the input vertex
enum topologika_result
topologika_query_persistence_with_statistics(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	double *out_persistence, struct topologika_query_statistics *out_statistics)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
			.region_index = vertex.region_index,
			.vertex_index = vertex.vertex_index,
			.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
			.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();
	// TODO(3/1/2021): we could extract visited regions directly from the visited set, but that requires
	//	knowledge of the internals of the set data structure; on the other hand, getting the correct set
	//	is less error prone
	hash_set_init(&out_statistics->visited_regions, 64);

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		struct topologika_vertex max = {0};
		topologika_query_component_max_terminate_early_with_statistics(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max, out_statistics);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				*out_persistence = v.value - item.value;
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistence = INFINITY;

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}




/////////////////////////// persistencebelow query ////////////////////////
// TODO(3/10/2021): do not return infinity when the persistence is not greater/eq the simplification threshold
enum topologika_result
topologika_query_persistencebelow(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double persistence_threshold,
	double *out_persistence)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
		.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
		.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		if (v.value - item.value >= persistence_threshold) {
			*out_persistence = INFINITY;
			return topologika_result_success;
		}

		struct topologika_vertex max = {0};
		topologika_query_component_max_terminate_early(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				*out_persistence = v.value - item.value;
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistence = INFINITY;

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}

enum topologika_result
topologika_query_persistencebelow_with_statistics(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double persistence_threshold,
	double *out_persistence, struct topologika_query_statistics *out_statistics)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
		.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
		.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();
	// TODO(3/1/2021): we could extract visited regions directly from the visited set, but that requires
	//	knowledge of the internals of the set data structure; on the other hand, getting the correct set
	//	is less error prone
	hash_set_init(&out_statistics->visited_regions, 64);

	// have to insert the initial region because we may early exit and not call the topologika_query_component_max_terminate_early_with_statistics
	//	which would insert the maxima's region
	hash_set_insert(&out_statistics->visited_regions, vertex.region_index);

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		if (v.value - item.value >= persistence_threshold) {
			*out_persistence = INFINITY;
			return topologika_result_success;
		}

		struct topologika_vertex max = {0};
		topologika_query_component_max_terminate_early_with_statistics(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max, out_statistics);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				*out_persistence = v.value - item.value;
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistence = INFINITY;

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}



enum topologika_result
topologika_query_persistencepairbelow(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double persistence_threshold,
	struct topologika_pair *out_persistencepair)
{
	assert(vertex.region_index < forest->merge_tree_count);

	struct component_max_result v = {
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
	};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){
		.value = domain->regions[vertex.region_index].data[vertex.vertex_index],
		.region_index = vertex.region_index,
		.vertex_index = vertex.vertex_index,
		.arc_index = forest->merge_trees[vertex.region_index].vertex_to_arc[vertex.vertex_index],
		.arc_region = vertex.region_index,
	});

	struct set *visited = set_create();

	struct component_max_result current_max = v;
	while (!topologika_priority_queue_is_empty(&pq)) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);

		// TODO: in debug mode make sure we the items are nonincreasing (same or decreasing)
		//assert(!topologika_is_above((struct component_max_result){item.value, item.vertex_index, item.region_index}, (struct component_max_result){current_item.value, current_item.vertex_index, current_item.region_index}));

		if (v.value - item.value >= persistence_threshold) {
			*out_persistencepair = (struct topologika_pair){
				.u = vertex,
				.s = topologika_bottom,
			};
			return topologika_result_success;
		}

		struct topologika_vertex max = {0};
		topologika_query_component_max_terminate_early(domain, forest, item.arc_region, item.arc_index, (struct component_max_result){.value = item.value, .region_index = item.region_index, .vertex_index = item.vertex_index}, visited, &pq, current_max, &max);
		if (max.vertex_index != TOPOLOGIKA_LOCAL_MAX) {
			struct component_max_result m = {
				.value = domain->regions[max.region_index].data[max.vertex_index],
				.region_index = max.region_index,
				.vertex_index = max.vertex_index,
			};
			current_max = m;

			if (topologika_is_above(m, v)) {
				*out_persistencepair = (struct topologika_pair){
					.u = vertex,
					.s = {.region_index = item.region_index, .vertex_index = item.vertex_index},
				};
				set_destroy(visited);
				free(pq.array);
				return topologika_result_success;
			}
		}
	}

	// global maximum
	*out_persistencepair = (struct topologika_pair){
		.u = vertex,
		.s = topologika_bottom,
	};

	set_destroy(visited);
	free(pq.array);
	return topologika_result_success;
}




////////////////////////// subtree max optimizations //////////////////////
// NOTE(3/5/2021): this optimization is pessimistic, because we do not split arcs using the reduced bridge set
//	end vertices
bool
topologika_is_above_local(topologika_data_t const *data, topologika_local_t v0, topologika_local_t v1)
{
	if (data[v0] == data[v1]) {
		return v0 > v1;
	}
	return data[v0] > data[v1];
}

void
topologika_precompute_subtree_max(struct topologika_domain const *domain, struct topologika_merge_forest *forest)
{
	for (int64_t region_i = 0; region_i < forest->merge_tree_count; region_i++) {
		struct topologika_merge_tree *tree = &forest->merge_trees[region_i];

		for (int64_t arc_i = 0; arc_i < tree->arc_count; arc_i++) {
			struct topologika_merge_tree_arc *arc = &tree->arcs[arc_i];

			topologika_local_t max = arc->max_vertex_id;

			bool all_internal = tree->reduced_bridge_set_counts[arc_i] == 0;
			for (int64_t i = 0; i < arc->child_count; i++) {
				assert(arc->children[i] < arc_i);
				if (tree->arcs[arc->children[i]].subtree_max_id == TOPOLOGIKA_LOCAL_MAX) {
					all_internal = false;
				}
				if (topologika_is_above_local(domain->regions[region_i].data, tree->arcs[arc->children[i]].subtree_max_id, max)) {
					max = tree->arcs[arc->children[i]].subtree_max_id;
				}
			}

			if (all_internal) {
				arc->subtree_max_id = max;
			} else {
				arc->subtree_max_id = TOPOLOGIKA_LOCAL_MAX;
			}
		}
	}
}




/////////////////////////// lazy evaluated components query ///////////////
enum topologika_result
query_components_region_internal_lazy(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t from_arc_id, topologika_local_t region_id, struct set *visited, struct worklist **todo,
	struct topologika_component **out_component)
{
	struct topologika_merge_tree const *tree = &forest->merge_trees[region_id];
	struct topologika_component *component = *out_component;

	// NOTE(11/25/2019): we do not use recursion to avoid stack overflow (the performance
	//	improvement is negligible for forest compared to a global tree)
	// TODO: stack allocator here, we should preallocate enough memory during forest
	//	construction that is sufficient for all temporary allocations during construction and queries
	int64_t stack_capacity = tree->arc_count;
	struct stack_item {
		topologika_local_t arc_id;
		topologika_local_t from_arc_id;
	} *stack = malloc(stack_capacity*sizeof *stack);
	assert(stack != NULL);
	int64_t stack_count = 0;
	stack[stack_count++] = (struct stack_item){.arc_id = arc_id, .from_arc_id = from_arc_id};
	while (stack_count != 0) {
		struct stack_item item = stack[--stack_count];

		struct topologika_merge_tree_arc const *arc = &tree->arcs[item.arc_id];

		// this part differs between the component max and component queries
		// callback(domain, forest, region_id, arc_id, context);

		bool crosses_threshold = !(arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold);

		// TODO(11/25/2019): we could store each segment as [region_id, count, local_id0, local_id1]
		// copy segmentation
		if (crosses_threshold) {
			// copy arc vertices above threshold
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];
				if (domain->regions[region_id].data[vertex_id] < threshold) {
					break;
				}

				/*
				if (component->count == component->capacity) {
					component->capacity *= 2;
					struct topologika_component *tmp = realloc(component, sizeof *component + component->capacity*sizeof *component->data);
					if (tmp == NULL) {
						free(stack);
						return topologika_error_out_of_memory;
					}
					component = tmp;
				}

				component->data[component->count++] = (struct topologika_vertex){
					.vertex_index = vertex_id,
					.region_index = region_id,
				};*/
				component->count++;
			}
		} else {
			// copy whole arc
			/*
			for (int64_t i = 0; i < tree->segmentation_counts[item.arc_id]; i++) {
				topologika_local_t vertex_id = tree->segmentation[tree->segmentation_offsets[item.arc_id] + i];

				// TODO: resize to the nearest power of two > component->count + vertex_count outside of the loop
				if (component->count == component->capacity) {
					component->capacity *= 2;
					struct topologika_component *tmp = realloc(component, sizeof *component + component->capacity*sizeof *component->data);
					if (tmp == NULL) {
						free(stack);
						return topologika_error_out_of_memory;
					}
					component = tmp;
				}

				component->data[component->count++] = (struct topologika_vertex){
					.vertex_index = vertex_id,
					.region_index = region_id,
				};
			}
			*/
			component->count += tree->segmentation_counts[item.arc_id];
		}


		// push all neighbors and reduced bridge set edges if needed to worklist
		for (int64_t i = 0; i < arc->child_count; i++) {
			if (arc->children[i] ==  item.from_arc_id) {
				continue;
			}
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->children[i],
				.from_arc_id = item.arc_id,
			};
		}

		if (arc->parent != TOPOLOGIKA_LOCAL_MAX && arc->parent != item.from_arc_id &&
			domain->regions[region_id].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
			assert(stack_count < stack_capacity);
			stack[stack_count++] = (struct stack_item){
				.arc_id = arc->parent,
				.from_arc_id = item.arc_id,
			};
		}

		int64_t worklist_count = (*todo)->count;
		for (int64_t i = 0; i < tree->reduced_bridge_set_counts[item.arc_id]; i++) {
			struct reduced_bridge_set_edge edge = tree->reduced_bridge_set->edges[tree->reduced_bridge_set_offsets[item.arc_id] + i];

			// TODO(11/25/2019): likely 2 cache misses
			if ((domain->regions[region_id].data[edge.local_id] < threshold) ||
				(domain->regions[edge.neighbor_region_id].data[edge.neighbor_local_id] < threshold)) {
				continue;
			}

			if ((*todo)->count == (*todo)->capacity) {
				(*todo)->capacity *= 2;
				struct worklist *tmp = realloc(*todo, sizeof *tmp + (*todo)->capacity*sizeof *tmp->items);
				if (tmp == NULL) {
					// TODO: free
					return topologika_error_out_of_memory;
				}
				*todo = tmp;
			}
			(*todo)->items[(*todo)->count++] = (struct worklist_item){
				.arc_id = forest->merge_trees[edge.neighbor_region_id].vertex_to_arc[edge.neighbor_local_id],
				.region_id = edge.neighbor_region_id,
			};
		}

		// we enqueued some bridge set edges, so we need to mark the arc as visited to
		//	avoid potentially jumping back to it through some bridge set edge; also
		//	crossing threshold arcs need to be marked too to avoid starting the extraction
		//	of component from them (component in a forest can have multiple crossed arcs
		//	in contrast to the global tree where it is always only 1 crossed arc)
		if (worklist_count != (*todo)->count || crosses_threshold) {
			set_insert(visited, item.arc_id, region_id);
		}
	}
	free(stack);

	*out_component = component;

	return topologika_result_success;
}


enum topologika_result
query_components_internal_lazy(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold, topologika_local_t arc_id, topologika_local_t region_id, struct set *visited,
	struct topologika_component **out_component)
{
	int64_t initial_component_capacity = 0;
	struct topologika_component *component = malloc(sizeof *component + initial_component_capacity*sizeof *component->data);
	if (component == NULL) {
		return topologika_error_out_of_memory;
	}
	component->count = 0;
	component->capacity = initial_component_capacity;

	// TODO: scratch alloc
	int64_t capacity = 1024;
	struct worklist *todo = malloc(sizeof *todo + capacity*sizeof *todo->items);
	if (todo == NULL) {
		free(component);
		return topologika_error_out_of_memory;
	}
	todo->capacity = capacity;
	todo->count = 0;

	enum topologika_result result = query_components_region_internal_lazy(domain, forest, threshold, arc_id, arc_id, region_id, visited, &todo, &component);
	if (result != topologika_result_success) {
		free(component);
		free(todo);
		return result;
	}

	for (int64_t i = 0; i < todo->count; i++) {
		if (set_contains(visited, todo->items[i].arc_id, todo->items[i].region_id)) {
			continue;
		}

		enum topologika_result result = query_components_region_internal_lazy(domain, forest, threshold, todo->items[i].arc_id, todo->items[i].arc_id, todo->items[i].region_id, visited, &todo, &component);
		if (result != topologika_result_success) {
			free(component);
			free(todo);
			return result;
		}
	}

	free(todo);

	*out_component = component;

	return topologika_result_success;
}


// TODO: the ***out_components is bit hairy (we need to return a pointer to array of pointers)
enum topologika_result
topologika_query_components_lazy(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_component ***out_components, int64_t *out_component_count)
{
	int64_t component_capacity = 16;
	struct topologika_component **components = malloc(component_capacity*sizeof *components);
	if (components == NULL) {
		return topologika_error_out_of_memory;
	}
	int64_t component_count = 0;

	struct set *visited = set_create(); // TODO: scratch allocator?
	assert(visited != NULL);

	for (int64_t region_index = 0; region_index < forest->merge_tree_count; region_index++) {
		struct topologika_merge_tree const *tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < tree->arc_count; arc_index++) {
			struct topologika_merge_tree_arc const *arc = &tree->arcs[arc_index];

			if (set_contains(visited, (topologika_local_t)arc_index, (topologika_local_t)region_index)) {
				continue;
			}

			// TODO: inline data values and profile
			if (domain->regions[region_index].data[arc->max_vertex_id] < threshold) {
				continue;
			}
			// the arc does not cross the threshold
			if (arc->parent != TOPOLOGIKA_LOCAL_MAX && domain->regions[region_index].data[tree->arcs[arc->parent].max_vertex_id] >= threshold) {
				continue;
			}

			struct topologika_component *component = NULL;
			enum topologika_result result = query_components_internal_lazy(domain, forest, threshold, (topologika_local_t)arc_index, (topologika_local_t)region_index, visited, &component);
			if (result != topologika_result_success) {
				*out_components = 0;
				set_destroy(visited);
				return result;
			}

			// empty component
			if (component != NULL && component->count == 0) {
				free(component);
				continue;
			}

			if (component_count == component_capacity) {
				component_capacity *= 2;
				struct topologika_component **tmp = realloc(components, component_capacity*sizeof *tmp);
				if (tmp == NULL) {
					free(components);
					set_destroy(visited);
					return topologika_error_out_of_memory;
				}
				components = tmp;
			}
			components[component_count++] = component;
		}
	}

	set_destroy(visited);

	*out_components = components;
	*out_component_count = component_count;

	return topologika_result_success;
}




////////////////////////// legacy conversion functions /////////////////////
struct topologika_vertex
topologika_global_index_to_vertex(int64_t const *dims, struct topologika_domain const *domain, int64_t global_vertex_index)
{
	int64_t global_position[] = {
		global_vertex_index%dims[0],
		global_vertex_index/dims[0]%dims[1],
		global_vertex_index/(dims[0]*dims[1]),
	};

	int64_t region_index = (global_position[0]/domain->region_dims[0]) +
		(global_position[1]/domain->region_dims[1])*domain->dims[0] +
		(global_position[2]/domain->region_dims[2])*domain->dims[0]*domain->dims[1];

	struct region const *region = &domain->regions[region_index];
	int64_t vertex_index = (global_position[0]%domain->region_dims[0]) +
		(global_position[1]%domain->region_dims[1])*region->dims[0] +
		(global_position[2]%domain->region_dims[2])*region->dims[0]*region->dims[1];

	assert(region_index < TOPOLOGIKA_LOCAL_MAX && vertex_index < TOPOLOGIKA_LOCAL_MAX);
	return (struct topologika_vertex){
		.region_index = (topologika_local_t)region_index,
		.vertex_index = (topologika_local_t)vertex_index,
	};
}


int64_t
topologika_vertex_to_global_index(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex vertex)
{
	struct region const *region = &domain->regions[vertex.region_index];
	int64_t global_position[] = {
		vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0],
		vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1],
		vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2],
	};

	return global_position[0] + global_position[1]*dims[0] + global_position[2]*dims[0]*dims[1];
}


void
topologika_vertices_to_global_indices(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex const *vertices, int64_t vertex_count,
		int64_t *indices)
{
	for (int64_t i = 0; i < vertex_count; i++) {
		struct topologika_vertex vertex = vertices[i];
		struct region const *region = &domain->regions[vertex.region_index];
		if (region->dims[0] == 64 && region->dims[1] == 64 && region->dims[2] == 64) {
			int64_t global_position[] = {
				vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0],
				vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1],
				vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2],
			};

			indices[i] = global_position[0] + global_position[1]*dims[0] + global_position[2]*dims[0]*dims[1];
		} else {
			int64_t global_position[] = {
				vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0],
				vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1],
				vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2],
			};

			indices[i] = global_position[0] + global_position[1]*dims[0] + global_position[2]*dims[0]*dims[1];
		}
	}
}


void
topologika_vertices_to_global_coordinates(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex const *vertices, int64_t vertex_count,
		int64_t *xs, int64_t *ys, int64_t *zs)
{
	for (int64_t i = 0; i < vertex_count; i++) {
		struct topologika_vertex vertex = vertices[i];
		struct region const *region = &domain->regions[vertex.region_index];
		if (region->dims[0] == 64 && region->dims[1] == 64 && region->dims[2] == 64) {
			xs[i] = vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0];
			ys[i] = vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1];
			zs[i] = vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2];
		} else {
			xs[i] = vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0];
			ys[i] = vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1];
			zs[i] = vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2];
		}
	}
}

void
topologika_vertices_to_global_coordinates16(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex const *vertices, int64_t vertex_count,
		int16_t *xs, int16_t *ys, int16_t *zs)
{
	for (int64_t i = 0; i < vertex_count; i++) {
		struct topologika_vertex vertex = vertices[i];
		struct region const *region = &domain->regions[vertex.region_index];
		if (region->dims[0] == 64 && region->dims[1] == 64 && region->dims[2] == 64) {
			xs[i] = (int16_t)(vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0]);
			ys[i] = (int16_t)(vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1]);
			zs[i] = (int16_t)(vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2]);
		} else {
			xs[i] = (int16_t)(vertex.vertex_index%region->dims[0] + (vertex.region_index%domain->dims[0])*domain->region_dims[0]);
			ys[i] = (int16_t)(vertex.vertex_index/region->dims[0]%region->dims[1] + (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1]);
			zs[i] = (int16_t)(vertex.vertex_index/(region->dims[0]*region->dims[1]) + (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2]);
		}
	}
}


void
topologika_vertices_to_global_indices_cache(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex const *vertices, int64_t vertex_count,
	int64_t *indices)
{
	struct region const *previous_region = NULL; 
	int64_t region_position[3];
	for (int64_t i = 0; i < vertex_count; i++) {
		struct topologika_vertex vertex = vertices[i];

		struct region const *region = &domain->regions[vertex.region_index];
		if (region != previous_region) {
			region_position[0] = (vertex.region_index%domain->dims[0])*domain->region_dims[0];
			region_position[1] = (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1];
			region_position[2] = (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2];
		}

		int64_t global_position[] = {
			vertex.vertex_index%region->dims[0] + (region_position[0]),
			vertex.vertex_index/region->dims[0]%region->dims[1] + (region_position[1]),
			vertex.vertex_index/(region->dims[0]*region->dims[1]) + (region_position[2]),
		};

		indices[i] = global_position[0] + global_position[1]*dims[0] + global_position[2]*dims[0]*dims[1];
	}
}



void
topologika_vertex_to_region_and_local_position(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex vertex,
	int64_t *region_position, int64_t *local_position)
{
	struct region const *region = &domain->regions[vertex.region_index];

	region_position[0] = (vertex.region_index%domain->dims[0])*domain->region_dims[0];
	region_position[1] = (vertex.region_index/domain->dims[0]%domain->dims[1])*domain->region_dims[1];
	region_position[2] = (vertex.region_index/(domain->dims[0]*domain->dims[1]))*domain->region_dims[2];

	local_position[0] = vertex.vertex_index%region->dims[0];
	local_position[1] = vertex.vertex_index/region->dims[0]%region->dims[1];
	local_position[2] = vertex.vertex_index/(region->dims[0]*region->dims[1]);
}


void
topologika_region_and_local_position_to_position(int64_t const *dims, struct topologika_domain const *domain, int64_t const *region_position, int64_t  const *local_position,
	int64_t *position)
{
	position[0] = region_position[0] + local_position[0];
	position[1] = region_position[1] + local_position[1];
	position[2] = region_position[2] + local_position[2];
}












// coordinate-based merge forest
/*
struct reduced_bridge_set {
	int64_t edge_count;
	struct reduced_bridge_set_edge edges[];
};

struct topologika_merge_tree2 {
	struct topologika_merge_tree_arc2 *arcs;
	int64_t arc_count;

	// indirect to compact segmentation representation that allows sublevel set extraction with single memcpy
	// includes the node's vertex_id
	// separate from arc as we do not require it during tree traversal
	// TODO: group offset and count into a struct?
	topologika_local_t *segmentation_offsets;
	topologika_local_t *segmentation_counts;

	topologika_local_t *segmentation;
	topologika_local_t *vertex_to_arc;

	struct reduced_bridge_set *reduced_bridge_set;
	// TODO: could these be topologika_local_t type and provably never overflow?
	int64_t *reduced_bridge_set_offsets;
	int64_t *reduced_bridge_set_counts;
};

struct topologika_merge_forest2 {
	int64_t merge_tree_count;
	struct topologika_merge_tree2 merge_trees[];
};




struct topologika_sort_vertex2 {
	topologika_data_t value;
	int8_t x, y, z;
};

int
topologika_decreasing_cmp2(void const *a, void const *b)
{
	struct topologika_sort_vertex2 const *v0 = b;
	struct topologika_sort_vertex2 const *v1 = a;

	if (v0->value == v1->value) {
		return (v0->index > v1->index) - (v0->index < v1->index);
	}
	return (v0->value < v1->value) ? -1 : 1;
}



enum topologika_result
compute_merge_tree2(struct region const *region, struct stack_allocator *stack_allocator, struct topologika_events *events, int64_t const *dims,
	struct topologika_merge_tree2 *tree)
{
	// TODO: runtime check that we deallocate the stack allocations in a reverse order; or even
	//	simpler would be just use a linear allocator and throw everything away at the end of the function
	// TODO: we could do a temporary large allocation and then copy to malloced memory; it is more robust with
	//	fewer failure points, but readibility suffers

	*tree = (struct topologika_merge_tree){0};
	struct stack_allocator initial_allocator = *stack_allocator;

	// TODO
	assert(dims[0] < 256 && dims[1] < 256 && dims[2] < 256);

	int64_t vertex_count = dims[0]*dims[1]*dims[2];

	// TODO: return pointer and then when we free data pass size
	struct topologika_sort_vertex2 *vertices = NULL;
	struct stack_allocation vertices_allocation = stack_allocator_alloc(stack_allocator, 8, vertex_count*sizeof *vertices);
	assert(vertices_allocation.ptr != NULL);
	vertices = vertices_allocation.ptr;

	int64_t arc_capacity = 32;
	{
		tree->arcs = malloc(arc_capacity*sizeof *tree->arcs);
		tree->segmentation_counts = malloc(arc_capacity*sizeof *tree->segmentation_counts);
		if (tree->arcs == NULL || tree->segmentation_counts == NULL) {
			goto out_of_memory;
		}
	}

	// inlining the data reduces number of indirections in the sort
	topologika_event_begin(events, topologika_event_color_green, "Init");
#if !defined(TOPOLOGIKA_THRESHOLD)
	int64_t offset = 0;
	for (int64_t k = 0; k < dims[2]; k++) {
		for (int64_t j = 0; j < dims[1]; j++) {
			for (int64_t i = 0; i < dims[0]; i++) {
				vertices[i] = (struct topologika_sort_vertex){
					.value = region->data[offset],
					.x = (int8_t)i,
					.y = (int8_t)j,
					.z = (int8_t)k,
				};
				offset++;
			}
		}
	}
#else
	int64_t thresholded_count = 0;
	for (int64_t i = 0; i < vertex_count; i++) {
		if (region->data[i] >= TOPOLOGIKA_THRESHOLD) {
			vertices[thresholded_count++] = (struct topologika_sort_vertex){.value = region->data[i], .index = (topologika_local_t)i};
		}
	}
	vertex_count = thresholded_count;
#endif
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Sort");
	qsort(vertices, vertex_count, sizeof *vertices, topologika_decreasing_cmp);
	topologika_event_end(events);

	int64_t vertex_count_ = dims[0]*dims[1]*dims[2];
	struct topologika_disjoint_set *components = NULL;
	struct stack_allocation components_allocation = stack_allocator_alloc(stack_allocator, 8, sizeof *components + vertex_count*sizeof *components->parent);
	if (components_allocation.ptr == NULL) {
		// TODO: free other memory
		return topologika_error_out_of_memory;
	}
	components = components_allocation.ptr;
	components->count = vertex_count;

	topologika_event_begin(events, topologika_event_color_green, "Alloc");
	tree->vertex_to_arc = malloc(vertex_count_*sizeof *tree->vertex_to_arc);
	if (tree->vertex_to_arc == NULL) {
		goto out_of_memory;
	}
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Clear");
	for (int64_t i = 0; i < vertex_count_; i++) {
		tree->vertex_to_arc[i] = TOPOLOGIKA_LOCAL_MAX;
	}
	topologika_event_end(events);

	topologika_event_begin(events, topologika_event_color_green, "Sweep");
	for (int64_t i = 0; i < vertex_count; i++) {
		topologika_local_t vertex_idx = vertices[i].index;
		int64_t vertex_position[] = {
			vertex_idx%dims[0],
			vertex_idx/dims[0]%dims[1],
			vertex_idx/(dims[0]*dims[1]),
		};

		struct topologika_disjoint_set_handle vertex_component = topologika_disjoint_set_invalid;

		// we exploit the fact that all children for a node are obtained in sequence, and thus
		//	can be appended in one go to the children buffer
		topologika_local_t children[neighbor_count];
		uint32_t child_count = 0;
		for (int64_t neighbor_i = 0; neighbor_i < neighbor_count; neighbor_i++) {
			int64_t neighbor_position[] = {
				vertex_position[0] + neighbors[neighbor_i][0],
				vertex_position[1] + neighbors[neighbor_i][1],
				vertex_position[2] + neighbors[neighbor_i][2],
			};

			// TODO: check if the compiler can do this cast optimization (e.g., fold x < 0 || x >= region->dims[0] => (uint64_t)x >= region->dims[0]
			if ((uint64_t)neighbor_position[0] >= (uint64_t)dims[0] || (uint64_t)neighbor_position[1] >= (uint64_t)dims[1] || (uint64_t)neighbor_position[2] >= (uint64_t)dims[2]) {
				continue;
			}

			int64_t neighbor_idx = neighbor_position[0] + neighbor_position[1]*dims[0] + neighbor_position[2]*dims[0]*dims[1];

			topologika_local_t neighbor_arc = tree->vertex_to_arc[neighbor_idx];
			if (neighbor_arc == TOPOLOGIKA_LOCAL_MAX) {
				continue;
			}

			struct topologika_disjoint_set_handle neighbor_component = topologika_disjoint_set_find(components, neighbor_arc);
			assert(neighbor_component.id != topologika_disjoint_set_invalid.id);

			if (vertex_component.id != neighbor_component.id) {
				children[child_count++] = neighbor_component.id; // same as the lowest arc in component

				// NOTE: assumes we get out neighbor_component label and thus regular vertex code below will work correctly
				if (vertex_component.id == topologika_disjoint_set_invalid.id) {
					vertex_component = neighbor_component;
				} else {
					// we always want lower component to point to higher component to get correct
					//	arc if neighbor was a regular vertex
					vertex_component = topologika_disjoint_set_union(components, neighbor_component, vertex_component);
				}
			}
		}

		// ensure we have enough space to store next critical point
		if (child_count != 1 && tree->arc_count == arc_capacity) {
			arc_capacity *= 2;
			struct topologika_merge_tree_arc* arcs = realloc(tree->arcs, arc_capacity * sizeof * tree->arcs);
			topologika_local_t* segmentation_counts = realloc(tree->segmentation_counts, arc_capacity * sizeof * tree->segmentation_counts);
			if (arcs == NULL || segmentation_counts == NULL) {
				goto out_of_memory;
			}
			tree->arcs = arcs;
			tree->segmentation_counts = segmentation_counts;
		}

		// maximum vertex
		if (child_count == 0) {
			tree->arcs[tree->arc_count] = (struct topologika_merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
			};
			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;
			topologika_disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			tree->arc_count++;

		// regular vertex
		} else if (child_count == 1) {
			tree->vertex_to_arc[vertex_idx] = children[0];
			tree->segmentation_counts[children[0]]++;

		// merge saddle vertex
		} else {
			tree->arcs[tree->arc_count] = (struct topologika_merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
				.child_count = child_count,
			};

			assert(child_count <= TOPOLOGIKA_COUNT_OF(tree->arcs[0].children));
			for (int64_t child_i = 0; child_i < child_count; child_i++) {
				tree->arcs[tree->arc_count].children[child_i] = children[child_i];
				tree->arcs[children[child_i]].parent = (topologika_local_t)tree->arc_count;
			}

			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;

			// disjoint set per arc
			struct topologika_disjoint_set_handle component = topologika_disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			topologika_disjoint_set_union(components, component, vertex_component); // NOTE: I think the order matters here
			tree->arc_count++;
		}
	}
	topologika_event_end(events);

	stack_allocator_free(stack_allocator, components_allocation);


	// build arc segmentation
	tree->segmentation = malloc(vertex_count*sizeof *tree->segmentation);
	if (tree->segmentation == NULL) {
		goto out_of_memory;
	}

	tree->segmentation_offsets = malloc(tree->arc_count*sizeof *tree->segmentation_offsets);
	if (tree->segmentation_offsets == NULL) {
		goto out_of_memory;
	}

	// prefix sum to figure out offsets into segmentation buffer (DFS? starting from root)
	topologika_event_begin(events, topologika_event_color_green, "Seg. offsets");
	for (int64_t i = tree->arc_count - 1, offset = 0; i >= 0; i--) {
		tree->segmentation_offsets[i] = (topologika_local_t)offset;
		offset += tree->segmentation_counts[i];
	}
	topologika_event_end(events);

	// TODO: we could do a depth-first traversal so the segmentation is linearized and any component
	//	of a subtree can be extracted with a memcpy
	// TODO: do we need the arc segmentation sorted? (now it is)
	// zero out so we can use it as offset in the assignment pass
	topologika_event_begin(events, topologika_event_color_green, "Seg. assign");
	memset(tree->segmentation_counts, 0x00, tree->arc_count*sizeof *tree->segmentation_counts);
	for (int64_t i = 0; i < vertex_count; i++) {
		topologika_local_t vertex_idx = vertices[i].index;
		topologika_local_t arc_id = tree->vertex_to_arc[vertex_idx];
		tree->segmentation[tree->segmentation_offsets[arc_id] + tree->segmentation_counts[arc_id]] = vertex_idx;
		tree->segmentation_counts[arc_id]++;
	}
	topologika_event_end(events);

	stack_allocator_free(stack_allocator, vertices_allocation);
	return topologika_result_success;

out_of_memory:
	free(tree->arcs);
	free(tree->segmentation);
	free(tree->segmentation_counts);
	free(tree->segmentation_offsets);
	free(tree->vertex_to_arc);
	*stack_allocator = initial_allocator;
	// TODO: more descriptive error message with location of code where allocation failed?
	return topologika_error_out_of_memory;
}



enum topologika_result
topologika_compute_merge_forest_from_grid2(topologika_data_t const *data, int64_t const *data_dims, int64_t const *region_dims,
	struct topologika_domain **out_domain, struct topologika_merge_forest2 **out_forest, double *out_construction_time)
{
	// TODO: we assume region_id <= 32 bits and vertex_id <= 32 bits
	assert(region_dims[0]*region_dims[1]*region_dims[2] <= ((int64_t)1 << (8*sizeof (topologika_local_t))));

	assert(region_dims[0] <= data_dims[0] && region_dims[1] <= data_dims[1] && region_dims[2] <= data_dims[2]);

	// heap-allocated toplevel pointers
	unsigned char *stack_allocators_memory = NULL;
	struct topologika_domain *domain = NULL;
	struct topologika_merge_forest *forest = NULL;
	struct topologika_events *events = NULL;
	struct stack_allocator *stack_allocators = NULL;

	// rounded up numbers of regions in each axis
	int64_t dims[] = {
		(data_dims[0] + region_dims[0] - 1)/region_dims[0],
		(data_dims[1] + region_dims[1] - 1)/region_dims[1],
		(data_dims[2] + region_dims[2] - 1)/region_dims[2],
	};
	int64_t region_count = dims[0]*dims[1]*dims[2];

	domain = malloc(sizeof *domain + region_count*sizeof *domain->regions);
	if (domain == NULL) {
		goto out_of_memory;
	}
	*domain = (struct topologika_domain){
		.data_dims = {data_dims[0], data_dims[1], data_dims[2]},
		.dims = {dims[0], dims[1], dims[2]},
		.region_dims = {region_dims[0], region_dims[1], region_dims[2]},
	};

	// decompose row-major array into regions
	int64_t region_index = 0;
	for (int64_t k = 0; k < data_dims[2]; k += region_dims[2]) {
		for (int64_t j = 0; j < data_dims[1]; j += region_dims[1]) {
			for (int64_t i = 0; i < data_dims[0]; i += region_dims[0]) {
				int64_t rdims[] = {
					(i + region_dims[0] > data_dims[0]) ? data_dims[0] - i : region_dims[0],
					(j + region_dims[1] > data_dims[1]) ? data_dims[1] - j : region_dims[1],
					(k + region_dims[2] > data_dims[2]) ? data_dims[2] - k : region_dims[2],
				};

				domain->regions[region_index] = (struct region){
					.data = malloc(rdims[0]*rdims[1]*rdims[2]*sizeof *domain->regions[region_index].data),
					.dims = {rdims[0], rdims[1], rdims[2]},
					.type = topologika_type_float,
				};
				if (domain->regions[region_index].data == NULL) {
					goto out_of_memory;
				}

				// copy the subset of global grid into the region
				int64_t position[] = {i, j, k};
				int64_t offset = 0;
				for (int64_t k = 0; k < rdims[2]; k++) {
					for (int64_t j = 0; j < rdims[1]; j++) {
						for (int64_t i = 0; i < rdims[0]; i++) {
							int64_t index = (i + position[0]) +
								(j + position[1])*data_dims[0] +
								(k + position[2])*data_dims[0]*data_dims[1];
							domain->regions[region_index].data[offset++] = data[index];
						}
					}
				}
				region_index++;
			}
		}
	}
	assert(region_index == region_count);

	int64_t event_capacity = (128 + 4)*region_count;
	if (record_events) {
		events = malloc(sizeof *events + event_capacity*sizeof *events->data);
		assert(events != NULL); // NOTE: we can assert because event recording is disabled by default
		events->count = 0;
		events->capacity = event_capacity;
	}

	int64_t start = topologika_usec_counter();
	topologika_event_begin(events, topologika_event_color_gray, "Compute forest");

	// create scratch allocators
	//int64_t thread_count = processor_count();
	int64_t thread_count = omp_get_max_threads();
	stack_allocators = malloc(thread_count*sizeof *stack_allocators);
	if (stack_allocators == NULL) {
		goto out_of_memory;
	}
	int64_t const vertex_count = region_dims[0]*region_dims[1]*region_dims[2];
	// we inline data during sort so extra space for it
	// TODO(9/17/2019): how much memory we need for bedges? (in Vis 2019 paper it was *8 instead of *16, but
	//	it is not enough on small region resolutions such as 4x4x4)
	// TODO: more exact formula
	int64_t max_region_dim = topologika_max(region_dims[0], topologika_max(region_dims[1], region_dims[2]));
	size_t size = (2*vertex_count*sizeof(struct topologika_sort_vertex)) + sizeof (struct bedge)*max_region_dim*max_region_dim*16;
	{
		stack_allocators_memory = calloc(thread_count, size);
		if (stack_allocators_memory == NULL) {
			goto out_of_memory;
		}
	}
	for (int64_t i = 0; i < thread_count; i++) {
		stack_allocator_init(&stack_allocators[i], size, &stack_allocators_memory[i*size]);
	}

	forest = malloc(sizeof *forest + region_count*sizeof *forest->merge_trees);
	if (forest == NULL) {
		goto out_of_memory;
	}
	*forest = (struct topologika_merge_forest){
		.merge_tree_count = region_count,
	};

	assert(region_count < INT_MAX);
	int64_t i;
#pragma omp parallel for schedule(dynamic)
	for (i = 0; i < region_count; i++) {
		struct region *region = &domain->regions[i];
		int tid = omp_get_thread_num();

		topologika_event_begin(events, topologika_event_color_green, "Compute MT");
		enum topologika_result result = compute_merge_tree(region, &stack_allocators[tid], events, region->dims, &forest->merge_trees[i]);
		// TODO(11/26/2019): how to bail from the parallel loop when computation fails
		assert(result == topologika_result_success);
		assert(stack_allocators[tid].offset == 0);
		topologika_event_end(events);

		topologika_event_begin(events, topologika_event_color_orange, "Compute RBS");
		result = compute_reduced_bridge_set(domain, i, &stack_allocators[tid], events, &forest->merge_trees[i].reduced_bridge_set);
		assert(result == topologika_result_success);
		assert(stack_allocators[tid].offset == 0);

		// build arc to bridge edges map
		{
			struct topologika_merge_tree *tree = &forest->merge_trees[i];
			struct reduced_bridge_set *set = tree->reduced_bridge_set;

			struct stack_allocation tmp_allocation = stack_allocator_alloc(&stack_allocators[tid], 8, set->edge_count*sizeof *set->edges);
			sort_edges(set->edge_count, set->edges, tmp_allocation.ptr, tree->vertex_to_arc);
			stack_allocator_free(&stack_allocators[tid], tmp_allocation);

			// TODO: merge into a single allocation? (more robust)
			tree->reduced_bridge_set_offsets = malloc(tree->arc_count*sizeof *tree->reduced_bridge_set_offsets);
			tree->reduced_bridge_set_counts = calloc(tree->arc_count, sizeof *tree->reduced_bridge_set_counts);
			assert(tree->reduced_bridge_set_offsets != NULL && tree->reduced_bridge_set_counts != NULL);
			for (int64_t edge_i = 0; edge_i < set->edge_count; edge_i++) {
				struct reduced_bridge_set_edge edge = set->edges[edge_i];
				topologika_local_t arc_id = tree->vertex_to_arc[edge.local_id];

				if (tree->reduced_bridge_set_counts[arc_id] == 0) {
					tree->reduced_bridge_set_offsets[arc_id] = edge_i;
				}
				tree->reduced_bridge_set_counts[arc_id]++;
			}
		}
		topologika_event_end(events);
	}
	free(stack_allocators);
	free(stack_allocators_memory);
	topologika_event_end(events);

	int64_t end = topologika_usec_counter();
	*out_construction_time = (end - start)*1e-6;

	if (record_events && events != NULL) {
		topologika_write_events("events.json", events);
		free(events);
	}

	*out_domain = domain;
	*out_forest = forest;
	return topologika_result_success;

out_of_memory:
	if (domain != NULL) {
		for (int64_t i = 0; i < region_count; i++) {
			free(domain->regions[i].data);
		}
	}
	free(domain);
	free(forest);
	free(events);
	free(stack_allocators_memory);
	free(stack_allocators);
	return topologika_error_out_of_memory;
}
*/











#endif


#endif
