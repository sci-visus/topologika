/*
topologika_merge_forest.h - public domain

Accelerated superlevel set analysis. Supports extraction of connected components,
finding maxima, and querying for representatives.

Requirements:
	- C99 (GCC >= 4.8, -std=gnu99, Clang >= 3.4, MSVC >= 1910)

References:
	- Toward Localized Topological Data Structures: Querying the Forest for the Tree.

Releases:
	- future
		- parallel query execution and scheduling
		- approximate analysis
	- 2020.x
		- add persistence simplified queries
		- add traverse component query (allows computation during the traversal)
		- support for domains that are not multiple of the region size
		- support adaptive resolution-precision grid (reduced bridge set is computed on edge stream)
	- 2019.11
		- initial port of VIS 2019 submission code with performance improvements and refactoring

Changes from the VIS code:
	- in code's simulation of simplicity, we use region_id and then vertex_id to break ties
		instead of using global row-major id
	- std::sort replaced by radix sort to avoid using the slow qsort available in C; also, qsort can't
		take context (portably) and thread local global variables are not ideal solution to pass context around
	- the bridge set computation does not require hardcoded neighbor region lookups
	- regions can have different sizes, thus the data dimensions do not need to be a multiple of the region size
	- overall, the performance has been improved (30% reduction of construction time, queries same); 1024^3 float
		data set construction takes about 10s on AMD 1950x
	- removed OpenMP dependency to simplify build process on Mac
*/

// TODO(3/19/2020): replace divisions by shifts when region_dims is power of two (or use JIT to generate the code with a compiler to keep the code readable)
// TODO(3/19/2020): the component query has large memory overhead, we either need to use generator or return it as a compact numpy array
// TODO(3/18/2020): use next pointer (linked-list) to store the arc's children indices instead of an array (more space efficient; should have the same performance)
// TODO(3/16/2020): we could sort the todos based on the region_id in componentmax and component query to improve cache locality (inspired by distributed forest implementation)
// TODO(3/3/2020): allow user specify number of threads to use for forest construction
// TODO(3/3/2020): OpenMP removal makes it harder to assign and pin threads to cores (and thus reduces
//	efficiency of the parallel execution because an OS may use hyperthreads or migrate threads around)
// TODO(2/27/2020): queries should not take the domain but just forest? (thus for many queries we could do away with the data),
//	this would require caching of function values at arc's highest vertices and reduced bridge set end vertices, but we avoid a cache
//	miss since when the arc/reduced bridge set edge is accessed we also pull the values
// TODO(1/7/2020): use edges instead of vertices for the reduced bridge set computation (simplifies support of Duong's grid)
// TODO(12/3/2019): try abstract the forest component traversal so it can be reused for component_max and maxima queries
// TODO(12/3/2019): pin threads or allow user to force thread pinning (similar to OMP_PLACES=cores and OMP_PROC_BIND=spread);
//	on windows pinning threads is problematic (windows does not like it)
// TODO(11/25/2019): topologika_local_t change to topologika_region_t for region indexing
//	(actually, enforcing same size of both is most convenient, because we can store them in the same array
//	in the components query)
//	on the other hand, we could not do 16 bit regions
// TODO(11/25/2019): store arc_id in the bridge set? (maybe even the function value so we do not need
//	the domain for traversal)
// TODO(11/19/2019): allow to save simplified field?
// TODO(11/15/2019): absan, ubsan, afl
// TODO(11/8/2019): when an out-of-memory error ocurrs, alongside the error report a lower bound on how much memory
//	would be required to build the merge forest or run the query

// NOTE(11/19/2019): compared to the old code, the reduced bridge set edges are sorted by region id
//	to break ties if local vertex ids and values are the same, thus the printout of edges per local tree
//	arc may have different order, but since the query correctness does not depend on the order
//	we get the same query output
// NOTE(11/17/2019): because we use radix sort, -0.0 < 0.0 (but in IEEE float they are equal); if it is a problem, we can convert
//	-0.0 to 0.0 before sorting


#if !defined(TOPOLOGIKA_MERGE_FOREST_H)
#define TOPOLOGIKA_MERGE_FOREST_H


#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



typedef float topologika_data_t;
typedef uint32_t topologika_local_t;
#define TOPOLOGIKA_LOCAL_MAX UINT32_MAX

struct topologika_domain;
struct topologika_merge_forest;

struct topologika_vertex {
	topologika_local_t region_index;
	topologika_local_t vertex_index;
};

enum topologika_result {
	topologika_result_success,
	topologika_error_no_output,
	topologika_error_out_of_memory,
};


// construction
enum topologika_result
topologika_compute_merge_forest_from_grid(topologika_data_t const *data, int64_t const *data_dims, int64_t const *region_dims,
	struct topologika_domain **out_domain, struct topologika_merge_forest **out_forest);


// queries
enum topologika_result
topologika_query_component_max(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_vertex *out_max_vertex);

enum topologika_result
topologika_query_maxima(struct topologika_domain const *domain, struct topologika_merge_forest const *forest,
	struct topologika_vertex **out_maxima, int64_t *out_maximum_count);

// TODO: are these style of arrays error prone? (compared to having a malloced pointer to vertices)
struct topologika_component {
	int64_t count;
	int64_t capacity;
	struct topologika_vertex data[];
};

enum topologika_result
topologika_query_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_component **out_component);

enum topologika_result
topologika_query_components(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, double threshold,
	struct topologika_component ***out_components, int64_t *out_component_count);


// legacy conversion functions from and to global coordinate space
struct topologika_vertex
topologika_global_index_to_vertex(int64_t const *dims, struct topologika_domain const *domain, int64_t global_vertex_index);

int64_t
topologika_vertex_to_global_index(int64_t const *dims, struct topologika_domain const *domain, struct topologika_vertex vertex);








#if defined(TOPOLOGIKA_MERGE_FOREST_IMPLEMENTATION)


// compile-time configuration
bool const record_events = false;


// macros
#define COUNT_OF(array) ((int64_t)(sizeof (array)/sizeof *(array)))


struct thread_context {
	struct topologika_domain *domain;
	struct topologika_merge_forest *forest;
	struct stack_allocator *stack_allocator;
	struct topologika_events *events;
	int64_t *work_offset;
	int64_t (*function)(struct thread_context *);
};


// platform-dependent code
#if defined(_WIN64)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int64_t
usec_counter(void)
{
	LARGE_INTEGER freq = {0};
	QueryPerformanceFrequency(&freq);
	LARGE_INTEGER time = {0};
	QueryPerformanceCounter(&time);
	return 1000000LL*time.QuadPart/freq.QuadPart;
}

int64_t
atomic_add(volatile int64_t *addend, int64_t value)
{
	return _InterlockedExchangeAdd64(addend, value);
}

int64_t
thread_id(void)
{
	return GetCurrentThreadId();
}

int64_t
processor_count(void)
{
	SYSTEM_INFO system_info = {0};
	GetSystemInfo(&system_info);
	return system_info.dwNumberOfProcessors;
}

DWORD WINAPI
topologika_thread_func(LPVOID parameter)
{
	struct thread_context *context = parameter;
	return (DWORD)context->function(context);
}
enum topologika_result
topologika_run(struct thread_context *contexts, int64_t thread_count, int64_t work_count)
{
	assert(thread_count > 0);

	HANDLE *handles = malloc(thread_count*sizeof *handles);
	if (handles == NULL) {
		return topologika_error_out_of_memory;
	}
	for (int64_t i = 0; i < thread_count; i++) {
		if (i + 1 == thread_count) {
			topologika_thread_func(&contexts[i]);
		} else {
			handles[i] = CreateThread(NULL, 0, topologika_thread_func, &contexts[i], 0, NULL);
		}
	}
	WaitForMultipleObjects((DWORD)(thread_count - 1), handles, TRUE, INFINITE);
	free(handles);

	return topologika_result_success;
}

// Linux
#elif __linux__

#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// TODO: do we need volatile?
int64_t
atomic_add(volatile int64_t *addend, int64_t value)
{
	return __sync_fetch_and_add(addend, value);
}

int64_t
usec_counter(void)
{
	struct timespec counter;
	clock_gettime(CLOCK_MONOTONIC, &counter);
	return (1000000000LL*counter.tv_sec + counter.tv_nsec)/1000LL;
}

int64_t
thread_id(void)
{
	return syscall(SYS_gettid);
}

int64_t
processor_count(void)
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

void *
topologika_thread_func(void *parameter)
{
	struct thread_context *context = parameter;
	return (void *)context->function(context);
}
enum topologika_result
topologika_run(struct thread_context *contexts, int64_t thread_count, int64_t work_count)
{
	assert(thread_count > 0);

	pthread_t *handles = malloc(thread_count*sizeof *handles);
	if (handles == NULL) {
		return topologika_error_out_of_memory;
	}
	for (int64_t i = 0; i < thread_count; i++) {
		if (i + 1 == thread_count) {
			topologika_thread_func(&contexts[i]);
		} else {
			int ret = pthread_create(&handles[i], NULL, topologika_thread_func, &contexts[i]);
			assert(ret == 0); // TODO: return error
		}
	}
	for (int64_t i = 0; i < thread_count - 1; i++) {
		pthread_join(handles[i], NULL);
	}
	free(handles);

	return topologika_result_success;
}

#elif defined(__APPLE__)

#include <pthread.h>
#include <time.h>
#include <unistd.h>

// TODO: do we need volatile?
int64_t
atomic_add(volatile int64_t *addend, int64_t value)
{
	return __sync_fetch_and_add(addend, value);
}

int64_t
usec_counter(void)
{
	struct timespec counter;
	clock_gettime(CLOCK_MONOTONIC, &counter);
	return (1000000000LL*counter.tv_sec + counter.tv_nsec)/1000LL;
}

int64_t
thread_id(void)
{
	uint64_t id = 0;
	pthread_threadid_np(NULL, &id);
	return (int64_t)id;
}

int64_t
processor_count(void)
{
	return sysconf(_SC_NPROCESSORS_ONLN);
}

void *
topologika_thread_func(void *parameter)
{
	struct thread_context *context = parameter;
	return (void *)context->function(context);
}
enum topologika_result
topologika_run(struct thread_context *contexts, int64_t thread_count, int64_t work_count)
{
	assert(thread_count > 0);

	pthread_t *handles = malloc(thread_count*sizeof *handles);
	if (handles == NULL) {
		return topologika_error_out_of_memory;
	}
	for (int64_t i = 0; i < thread_count; i++) {
		if (i + 1 == thread_count) {
			topologika_thread_func(&contexts[i]);
		} else {
			int ret = pthread_create(&handles[i], NULL, topologika_thread_func, &contexts[i]);
			assert(ret == 0); // TODO: return error
		}
	}
	for (int64_t i = 0; i < thread_count - 1; i++) {
		pthread_join(handles[i], NULL);
	}
	free(handles);

	return topologika_result_success;
}

#else
#error "only Windows 64-bit, Mac OS X, and Linux are supported"
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

	int64_t offset = atomic_add(&events->count, 1);
	assert(offset < events->capacity);

	struct topologika_event event = {
		.color = color,
		.ph = 'B',
		.ts = usec_counter(),
		.tid = (int16_t)thread_id(),
	};
	int written = snprintf(event.name, sizeof event.name, "%s", name);
	assert(written >= 0 && written < COUNT_OF(event.name));
	events->data[offset] = event;
}

void
topologika_event_end(struct topologika_events *events)
{
	if (!record_events) {
		return;
	}

	int64_t offset = atomic_add(&events->count, 1);
	assert(offset < events->capacity);

	struct topologika_event event = {
		.ph = 'E',
		.ts = usec_counter(),
		.tid = (int16_t)thread_id(),
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



// 6 subdivision
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
enum {neighbor_count = COUNT_OF(neighbors)};


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

struct merge_tree_arc {
	// TODO: store function value?
	topologika_local_t max_vertex_id; // TODO: could read from segmentation
	// TODO: figure out the common case, indirect for the uncommon case
	topologika_local_t children[neighbor_count]; // TODO: indirect to a children array
	topologika_local_t child_count;
	topologika_local_t parent;
};

struct merge_tree {
	struct merge_tree_arc *arcs;
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
	struct merge_tree merge_trees[];
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
struct disjoint_set {
	int64_t count;
	topologika_local_t parent[];
};

struct disjoint_set_handle {
	topologika_local_t id;
};
struct disjoint_set_handle disjoint_set_invalid = {TOPOLOGIKA_LOCAL_MAX};

struct disjoint_set_handle
disjoint_set_mk(struct disjoint_set *ds, topologika_local_t id)
{
	assert(id < ds->count);
	ds->parent[id] = id;
	return (struct disjoint_set_handle){id};
}

struct disjoint_set_handle
disjoint_set_find_no_check(struct disjoint_set *ds, topologika_local_t id)
{
	if (ds->parent[id] == id) {
		return (struct disjoint_set_handle){id};
	}
	ds->parent[id] = disjoint_set_find_no_check(ds, ds->parent[id]).id;
	return (struct disjoint_set_handle){ds->parent[id]};
}

struct disjoint_set_handle
disjoint_set_find(struct disjoint_set *ds, topologika_local_t id)
{
	assert(id < ds->count);
	if (ds->parent[id] == TOPOLOGIKA_LOCAL_MAX) {
		return disjoint_set_invalid;
	}
	return disjoint_set_find_no_check(ds, id);
}

struct disjoint_set_handle
disjoint_set_union(struct disjoint_set *ds, struct disjoint_set_handle a, struct disjoint_set_handle b)
{
	ds->parent[b.id] = a.id;
	return a;
}







// ascending order
struct inlined_vertex_uint16 {
	uint16_t value;
	topologika_local_t index;
};
struct inlined_vertex_float {
	float value;
	topologika_local_t index;
};
struct inlined_vertex_double {
	double value;
	topologika_local_t index;
};

void
radix_sort_float(int64_t count, struct inlined_vertex_float *in, struct inlined_vertex_float *out, int shift)
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
sort_vertices_float(int64_t vertex_count, struct inlined_vertex_float *vertices, struct inlined_vertex_float *tmp)
{
	radix_sort_float(vertex_count, vertices, tmp, 0);
	radix_sort_float(vertex_count, tmp, vertices, 8);
	radix_sort_float(vertex_count, vertices, tmp, 16);
	radix_sort_float(vertex_count, tmp, vertices, 24);
}




enum topologika_result
compute_merge_tree(struct region const *region, struct stack_allocator *stack_allocator, struct topologika_events *events, int64_t const *dims,
	struct merge_tree *tree)
{
	// TODO: runtime check that we deallocate the stack allocations in a reverse order; or even
	//	simpler would be just use a linear allocator and throw everything away at the end of the function
	// TODO: we could do a temporary large allocation and then copy to malloced memory; it is more robust with
	//	fewer failure points, but readibility suffers

	*tree = (struct merge_tree){0};
	struct stack_allocator initial_allocator = *stack_allocator;

	int64_t const vertex_count = dims[0]*dims[1]*dims[2];

	// TODO: return pointer and then when we free data pass size
	struct inlined_vertex_float *vertices = NULL;
	struct stack_allocation vertices_allocation = stack_allocator_alloc(stack_allocator, 8, vertex_count*sizeof *vertices);
	assert(vertices_allocation.ptr != NULL);
	vertices = vertices_allocation.ptr;

	int64_t arc_capacity = 1024;
	{
		tree->arcs = malloc(arc_capacity*sizeof *tree->arcs);
		tree->segmentation_counts = malloc(arc_capacity*sizeof *tree->segmentation_counts);
		if (tree->arcs == NULL || tree->segmentation_counts == NULL) {
			goto out_of_memory;
		}
	}

	// inlining the data reduces number of indirections in the sort
	topologika_event_begin(events, topologika_event_color_green, "Init");
	for (int64_t i = 0; i < vertex_count; i++) {
		vertices[i] = (struct inlined_vertex_float){.value = region->data[i], .index = (topologika_local_t)i};
	}
	topologika_event_end(events);

	struct stack_allocation tmp_allocation = stack_allocator_alloc(stack_allocator, 8, vertex_count*sizeof *vertices);
	topologika_event_begin(events, topologika_event_color_green, "Sort");
	sort_vertices_float(vertex_count, vertices, tmp_allocation.ptr);
	topologika_event_end(events);
	stack_allocator_free(stack_allocator, tmp_allocation);


	struct disjoint_set *components = NULL;
	struct stack_allocation components_allocation = stack_allocator_alloc(stack_allocator, 8, sizeof *components + vertex_count*sizeof *components->parent);
	if (components_allocation.ptr == NULL) {
		// TODO: free other memory
		return topologika_error_out_of_memory;
	}
	components = components_allocation.ptr;
	components->count = vertex_count;

	tree->vertex_to_arc = malloc(vertex_count*sizeof *tree->vertex_to_arc);
	if (tree->vertex_to_arc == NULL) {
		goto out_of_memory;
	}
	memset(tree->vertex_to_arc, 0xFF, vertex_count*sizeof *tree->vertex_to_arc);


	topologika_event_begin(events, topologika_event_color_green, "Sweep");
	// descending order
	for (int64_t i = vertex_count - 1; i >= 0; i--) {
		topologika_local_t vertex_idx = vertices[i].index;
		int64_t vertex_position[] = {
			vertex_idx%dims[0],
			vertex_idx/dims[0]%dims[1],
			vertex_idx/(dims[0]*dims[1]),
		};

		struct disjoint_set_handle vertex_component = disjoint_set_invalid;

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

			struct disjoint_set_handle neighbor_component = disjoint_set_find(components, neighbor_arc);
			assert(neighbor_component.id != disjoint_set_invalid.id);

			if (vertex_component.id != neighbor_component.id) {
				children[child_count++] = neighbor_component.id; // same as the lowest arc in component

				// NOTE: assumes we get out neighbor_component label and thus regular vertex code below will work correctly
				if (vertex_component.id == disjoint_set_invalid.id) {
					vertex_component = neighbor_component;
				} else {
					// we always want lower component to point to higher component to get correct
					//	arc if neighbor was a regular vertex
					vertex_component = disjoint_set_union(components, neighbor_component, vertex_component);
				}
			}
		}

		// ensure we have enough space to store next critical point
		if (child_count != 1 && tree->arc_count == arc_capacity) {
			arc_capacity *= 2;
			struct merge_tree_arc* arcs = realloc(tree->arcs, arc_capacity * sizeof * tree->arcs);
			topologika_local_t* segmentation_counts = realloc(tree->segmentation_counts, arc_capacity * sizeof * tree->segmentation_counts);
			if (arcs == NULL || segmentation_counts == NULL) {
				goto out_of_memory;
			}
			tree->arcs = arcs;
			tree->segmentation_counts = segmentation_counts;
		}

		// maximum vertex
		if (child_count == 0) {
			tree->arcs[tree->arc_count] = (struct merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
			};
			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;
			disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			tree->arc_count++;

		// regular vertex
		} else if (child_count == 1) {
			tree->vertex_to_arc[vertex_idx] = children[0];
			tree->segmentation_counts[children[0]]++;

		// merge saddle vertex
		} else {
			tree->arcs[tree->arc_count] = (struct merge_tree_arc){
				.max_vertex_id = vertex_idx,
				.parent = TOPOLOGIKA_LOCAL_MAX,
				.child_count = child_count,
			};

			assert(child_count <= COUNT_OF(tree->arcs[0].children));
			for (int64_t child_i = 0; child_i < child_count; child_i++) {
				tree->arcs[tree->arc_count].children[child_i] = children[child_i];
				tree->arcs[children[child_i]].parent = (topologika_local_t)tree->arc_count;
			}

			tree->segmentation_counts[tree->arc_count] = 1;
			tree->vertex_to_arc[vertex_idx] = (topologika_local_t)tree->arc_count;

			// disjoint set per arc
			struct disjoint_set_handle component = disjoint_set_mk(components, (topologika_local_t)tree->arc_count);
			disjoint_set_union(components, component, vertex_component); // NOTE: I think the order matters here
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
	// descending order
	for (int64_t i = vertex_count - 1; i >= 0; i--) {
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
				vertices[vertex_count++] = (struct vertex){region->data[id], topologika_local_to_global_id(domain, id, region_id)};
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
		for (int64_t i = 0; i < COUNT_OF(reduced_bridge_sets); i++) {
			if (reduced_bridge_sets[i] != NULL) {
				count += reduced_bridge_sets[i]->edge_count;
			}
		}

		reduced_bridge_set = malloc(sizeof *reduced_bridge_set + count*sizeof *reduced_bridge_set->edges);
		assert(reduced_bridge_set != NULL);
		reduced_bridge_set->edge_count = 0;

		for (int64_t i = 0; i < COUNT_OF(reduced_bridge_sets); i++) {
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





int64_t
compute_region(struct thread_context *context)
{
	while (true) {
		int64_t i = atomic_add(context->work_offset, 1);
		if (i >= context->forest->merge_tree_count) {
			return 0;
		}

		struct region *region = &context->domain->regions[i];

		topologika_event_begin(context->events, topologika_event_color_green, "Compute MT");
		enum topologika_result result = compute_merge_tree(region, context->stack_allocator, context->events, region->dims, &context->forest->merge_trees[i]);
		// TODO(11/26/2019): how to bail from the parallel loop when computation fails
		assert(result == topologika_result_success);
		assert(context->stack_allocator->offset == 0);
		topologika_event_end(context->events);

		topologika_event_begin(context->events, topologika_event_color_orange, "Compute RBS");
		result = compute_reduced_bridge_set(context->domain, i, context->stack_allocator, context->events, &context->forest->merge_trees[i].reduced_bridge_set);
		assert(result == topologika_result_success);
		assert(context->stack_allocator->offset == 0);

		// build arc to bridge edges map
		{
			struct merge_tree *tree = &context->forest->merge_trees[i];
			struct reduced_bridge_set *set = tree->reduced_bridge_set;

			struct stack_allocation tmp_allocation = stack_allocator_alloc(context->stack_allocator, 8, set->edge_count*sizeof *set->edges);
			sort_edges(set->edge_count, set->edges, tmp_allocation.ptr, tree->vertex_to_arc);
			stack_allocator_free(context->stack_allocator, tmp_allocation);

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
		topologika_event_end(context->events);
	}
}


enum topologika_result
topologika_compute_merge_forest_from_grid(topologika_data_t const *data, int64_t const *data_dims, int64_t const *region_dims,
	struct topologika_domain **out_domain, struct topologika_merge_forest **out_forest)
{
	// TODO: we assume region_id <= 32 bits and vertex_id <= 32 bits
	assert(region_dims[0]*region_dims[1]*region_dims[2] <= ((int64_t)1 << (8*sizeof (topologika_local_t))));

	// heap-allocated toplevel pointers
	unsigned char *stack_allocators_memory = NULL;
	struct topologika_domain *domain = NULL;
	struct topologika_merge_forest *forest = NULL;
	struct topologika_events *events = NULL;
	struct stack_allocator *stack_allocators = NULL;
	struct thread_context *contexts = NULL;

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
								(k + position[2])*data_dims[1]*data_dims[2];
							domain->regions[region_index].data[offset++] = data[index];
						}
					}
				}
				region_index++;
			}
		}
	}
	assert(region_index == region_count);

	int64_t event_capacity = 128*region_count;
	if (record_events) {
		events = malloc(sizeof *events + event_capacity*sizeof *events->data);
		assert(events != NULL); // NOTE: we can assert because event recording is disabled by default
		events->count = 0;
		events->capacity = event_capacity;
	}

	uint64_t start = usec_counter();
	topologika_event_begin(events, topologika_event_color_gray, "Compute forest");

	// create scratch allocators
	int64_t thread_count = processor_count();
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
	size_t size = (2*vertex_count*sizeof(struct inlined_vertex_float)) + sizeof (struct bedge)*max_region_dim*max_region_dim*16;
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

	int64_t work_offset = 0;
	contexts = malloc(thread_count*sizeof *contexts);
	if (contexts == NULL) {
		goto out_of_memory;
	}
	for (int i = 0; i < thread_count; i++) {
		contexts[i] = (struct thread_context){
			.domain = domain,
			.forest = forest,
			.stack_allocator = &stack_allocators[i],
			.events = events,
			.work_offset = &work_offset,
			.function = compute_region,
		};
	}
	// TODO: each computation should write result to a buffer and we should check all succeeded here
	enum topologika_result result = topologika_run(contexts, thread_count, forest->merge_tree_count);
	if (result != topologika_result_success) {
		// TODO: free memory
		return result;
	}
	free(contexts);
	free(stack_allocators);
	free(stack_allocators_memory);
	topologika_event_end(events);

	uint64_t end = usec_counter();
	printf("time %f ms\n", (end - start)*1e-3);

	if (record_events && events != NULL) {
		topologika_write_events("events.json", events);
		free(events);
	}

	*out_domain = domain;
	*out_forest = forest;
	return topologika_result_success;

out_of_memory:
	// TODO: share parts with dealloc in Python
	if (domain != NULL) {
		for (int64_t i = 0; i < region_count; i++) {
			free(domain->regions[i].data);
		}
	}
	free(domain);
	free(forest);
	free(contexts);
	free(events);
	free(stack_allocators_memory);
	free(stack_allocators);
	return topologika_error_out_of_memory;
}


////////////////////// component max query //////////////////////////////////////

struct hash_set {
	uint64_t count;
	uint64_t capacity;
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
	hash_set_init(&set->data, 1024);
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
	struct merge_tree const *tree = &forest->merge_trees[region_id];
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

		struct merge_tree_arc const *arc = &tree->arcs[item.arc_id];

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
// TODO(2/27/2020): should no result be an error or success (bottom)
enum topologika_result
topologika_query_component_max(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_vertex *out_max_vertex)
{
	assert(domain != NULL && forest != NULL);

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
// TODO: take in thresholds? bounding boxes?
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
		struct merge_tree const *merge_tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < merge_tree->arc_count; arc_index++) {
			struct merge_tree_arc const *arc = &merge_tree->arcs[arc_index];
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
	struct merge_tree const *tree = &forest->merge_trees[region_id];
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

		struct merge_tree_arc const *arc = &tree->arcs[item.arc_id];

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
		struct merge_tree const *tree = &forest->merge_trees[region_index];

		for (int64_t arc_index = 0; arc_index < tree->arc_count; arc_index++) {
			struct merge_tree_arc const *arc = &tree->arcs[arc_index];

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




////////////////////////// component query /////////////////////////
enum topologika_result
topologika_query_component(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex, double threshold,
	struct topologika_component **out_component)
{
	assert(vertex.region_index >= 0 && vertex.region_index < forest->merge_tree_count);
	struct merge_tree const *tree = &forest->merge_trees[vertex.region_index];

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




////////////////////////// persistence query //////////////////////////////
struct topologika_item {
	double value;
	topologika_local_t region_index, vertex_index;
	topologika_local_t arc_index;
};

bool
is_above_(struct topologika_item a, struct topologika_item b)
{
	if (a.value == b.value) {
		assert(false);
	}
	return a.value > b.value;
}

struct topologika_priority_queue {
	struct topologika_item *array;
	int64_t count, capacity;
};


void
topologika_priority_queue_init(struct topologika_priority_queue *pq)
{
	pq->count = 1; // NOTE(3/10/2020): we index starting from 1 to simplify parent index computation
	pq->capacity = 8;
	pq->array = malloc(pq->capacity*sizeof *pq->array);
	assert(pq->array != NULL);
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
	assert(pq->count < 1);

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
topologika_query_persistence(struct topologika_domain const *domain, struct topologika_merge_forest const *forest, struct topologika_vertex vertex,
	double *out_persistence)
{
	printf("\n\nQUERY r %d v %d\n", vertex.region_index, vertex.vertex_index);

	// TODO(2/28/2020): check if we got a regular vertex as input and return 0 early? (probably not worth the extra
	//	complexity to avoid one ComponentMax query)

	bool visited[16][16] = {0};

	struct topologika_priority_queue pq = {0};
	topologika_priority_queue_init(&pq);

	topologika_priority_queue_enqueue(&pq, (struct topologika_item){domain->regions[vertex.region_index].data[vertex.vertex_index], vertex.region_index, vertex.vertex_index, -1});

	while (pq.count > 1) {
		struct topologika_item item = topologika_priority_queue_dequeue(&pq);
		visited[item.region_index][item.vertex_index] = true;

		struct topologika_vertex v = {.vertex_index = item.vertex_index, .region_index = item.region_index};
		printf("compmax query: region %d, local %d, threshold %f\n", v.region_index, v.vertex_index, domain->regions[v.region_index].data[v.vertex_index]);
		struct topologika_vertex v_star = {0};
		enum topologika_result result = topologika_query_component_max(domain, forest, v, domain->regions[v.region_index].data[v.vertex_index], &v_star);
		assert(result == topologika_result_success);

		if (vertex.region_index != v_star.region_index || vertex.vertex_index != v_star.vertex_index) {
			printf("max r %d v %d\n", v_star.region_index, v_star.vertex_index);
			*out_persistence = (double)domain->regions[vertex.region_index].data[vertex.vertex_index] - (double)domain->regions[v.region_index].data[v.vertex_index];
			free(pq.array); // TODO
			return topologika_result_success;
		}

		topologika_local_t arc_index = forest->merge_trees[v.region_index].vertex_to_arc[v.vertex_index];
		struct merge_tree_arc const *arc = &forest->merge_trees[v.region_index].arcs[arc_index];
		if (arc->parent != TOPOLOGIKA_LOCAL_MAX) {
			struct topologika_vertex vv = {
				.vertex_index = forest->merge_trees[v.region_index].arcs[arc->parent].max_vertex_id,
				.region_index = v.region_index,
			};
			topologika_priority_queue_enqueue(&pq, (struct topologika_item){domain->regions[vv.region_index].data[vv.vertex_index], vv.region_index, vv.vertex_index, -1});
		}

		// NOTE(3/3/2020): enqueue all reduced bridge set end vertices for now
		for (int64_t i = 0; i < forest->merge_trees[v.region_index].reduced_bridge_set_counts[arc_index]; i++) {
			struct reduced_bridge_set_edge e = forest->merge_trees[v.region_index].reduced_bridge_set->edges[forest->merge_trees[v.region_index].reduced_bridge_set_offsets[arc_index] + i];
			printf("e r %d v %d, r %d v %d\n", v.region_index, e.local_id, e.neighbor_region_id, e.neighbor_local_id);
			// TODO: sos
			topologika_data_t value = domain->regions[v.region_index].data[v.vertex_index];
			topologika_data_t value0 = domain->regions[v.region_index].data[e.local_id];
			topologika_data_t value1 = domain->regions[e.neighbor_region_id].data[e.neighbor_local_id];
			if (value0 < value1) {
				// we do not want to enqueue bridge set edges above the current value, because the current arc
				//	can have bridge set edge above the 'value' that is in the other branch of the merge saddle,
				//	and then the component max of that branch will return different maximum (because we
				//	always use the threshold at the vertex, if we carried the 'min' of the thresholds, then
				//	it would work correctly without the 'value0 < value' check); of course, the componentmax
				//	cache would be less useful)
				if (!visited[v.region_index][e.local_id] && value0 < value)
					topologika_priority_queue_enqueue(&pq, (struct topologika_item){value0, v.region_index, e.local_id, -1});
			}
			else {
				if (!visited[e.neighbor_region_id][e.neighbor_local_id] && value1 < value)
					topologika_priority_queue_enqueue(&pq, (struct topologika_item){value1, e.neighbor_region_id, e.neighbor_local_id, -1});
			}
		}
	}

	*out_persistence = INFINITY;

	free(pq.array); // TODO
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


#endif


#endif