// reference implementation of queries directly on a grid

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>


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
enum {neighbor_count = sizeof neighbors/sizeof *neighbors};


void
topologika_reference_query_component(data_t const *data, int64_t const *data_dims, int64_t const *region_dims, int64_t global_vertex, double threshold, int64_t **out_vertices, int64_t *out_vertex_count)
{
	if (data[global_vertex] < threshold) {
		*out_vertices = NULL;
		*out_vertex_count = 0;
		return;
	}

	// round-up the number of regions on each axis
	int64_t dims[] = {
		(data_dims[0] + region_dims[0] - 1)/region_dims[0],
		(data_dims[1] + region_dims[1] - 1)/region_dims[1],
		(data_dims[2] + region_dims[2] - 1)/region_dims[2],
	};

	// TODO: support not a region multiple
	assert(data_dims[0]%region_dims[0] == 0 && data_dims[1]%region_dims[1] == 0 && data_dims[2]%region_dims[2] == 0);

	int64_t count = data_dims[0]*data_dims[1]*data_dims[2];

	bool *visited = calloc(count, sizeof *visited);
	assert(visited != NULL);

	int64_t *todo = malloc(count*sizeof *todo);
	assert(todo != NULL);
	int64_t todo_count = 0;

	int64_t vertex_capacity = 1024;
	int64_t vertex_count = 0;
	int64_t *vertices = malloc(vertex_capacity*sizeof *vertices);
	assert(vertices != NULL);

	// depth-first search
	todo[todo_count++] = global_vertex;
	visited[global_vertex] = true;
	while (todo_count != 0) {
		int64_t vertex = todo[--todo_count];

		if (vertex_count == vertex_capacity) {
			vertex_capacity *= 2;
			int64_t *tmp = realloc(vertices, vertex_capacity*sizeof *tmp);
			assert(tmp != NULL);
			vertices = tmp;
		}
		vertices[vertex_count++] = vertex;

		int64_t global_position[] = {
			vertex%data_dims[0],
			vertex/data_dims[0]%data_dims[1],
			vertex/(data_dims[0]*data_dims[1]),
		};
		int64_t local_idx = (global_position[0]%region_dims[0]) +
			(global_position[1]%region_dims[1])*region_dims[0] +
			(global_position[2]%region_dims[2])*region_dims[0]*region_dims[1];
		int64_t region_idx = (global_position[0]/region_dims[0]) +
			(global_position[1]/region_dims[1])*dims[0] +
			(global_position[2]/region_dims[2])*dims[0]*dims[1];

		for (int64_t n_i = 0; n_i < neighbor_count; n_i++) {
			int64_t neighbor_position[] = {
				global_position[0] + neighbors[n_i][0],
				global_position[1] + neighbors[n_i][1],
				global_position[2] + neighbors[n_i][2],
			};

			if (neighbor_position[0] < 0 || neighbor_position[0] >= data_dims[0] ||
				neighbor_position[1] < 0 || neighbor_position[1] >= data_dims[1] ||
				neighbor_position[2] < 0 || neighbor_position[2] >= data_dims[2]) {
				continue;
			}

			int64_t neighbor_global_idx = neighbor_position[0] +
				neighbor_position[1]*data_dims[0] +
				neighbor_position[2]*data_dims[0]*data_dims[1];

			if (visited[neighbor_global_idx]) {
				continue;
			}
			if (data[neighbor_global_idx] >= threshold) {
				visited[neighbor_global_idx] = true;
				assert(todo_count < count);
				todo[todo_count++] = neighbor_global_idx;
			}
		}
	}

	*out_vertices = vertices;
	*out_vertex_count = vertex_count;
}



void
topologika_reference_query_maxima(data_t const *data, int64_t const *data_dims, int64_t const *region_dims, int64_t **out_maxima, int64_t *out_maximum_count)
{
	// round-up the number of regions on each axis
	int64_t dims[] = {
		(data_dims[0] + region_dims[0] - 1)/region_dims[0],
		(data_dims[1] + region_dims[1] - 1)/region_dims[1],
		(data_dims[2] + region_dims[2] - 1)/region_dims[2],
	};

	// TODO: support not a region multiple
	assert(data_dims[0]%region_dims[0] == 0 && data_dims[1]%region_dims[1] == 0 && data_dims[2]%region_dims[2] == 0);

	int64_t maximum_capacity = 1024;
	int64_t maximum_count = 0;
	int64_t *maxima = malloc(maximum_capacity*sizeof *maxima);
	assert(maxima != NULL);

	int64_t global_idx = 0;
	for (int64_t k = 0; k < data_dims[2]; k++) {
		for (int64_t j = 0; j < data_dims[1]; j++) {
			for (int64_t i = 0; i < data_dims[0]; i++) {
				int64_t local_idx = (i%region_dims[0]) +
					(j%region_dims[1])*region_dims[0] +
					(k%region_dims[2])*region_dims[0]*region_dims[1];
				int64_t region_idx = (i/region_dims[0]) +
					(j/region_dims[1])*dims[0] +
					(k/region_dims[2])*dims[0]*dims[1];

				bool all_lower = true;
				for (int64_t n_i = 0; n_i < neighbor_count; n_i++) {
					int64_t neighbor_position[] = {
						i + neighbors[n_i][0],
						j + neighbors[n_i][1],
						k + neighbors[n_i][2],
					};

					if (neighbor_position[0] < 0 || neighbor_position[0] >= data_dims[0] ||
						neighbor_position[1] < 0 || neighbor_position[1] >= data_dims[1] ||
						neighbor_position[2] < 0 || neighbor_position[2] >= data_dims[2]) {
						continue;
					}

					int64_t neighbor_global_idx = neighbor_position[0] +
						neighbor_position[1]*data_dims[0] +
						neighbor_position[2]*data_dims[0]*data_dims[1];
					int64_t neighbor_local_idx = (neighbor_position[0]%region_dims[0]) +
						(neighbor_position[1]%region_dims[1])*region_dims[0] +
						(neighbor_position[2]%region_dims[2])*region_dims[0]*region_dims[1];
					int64_t neighbor_region_idx = (neighbor_position[0]/region_dims[0]) +
						(neighbor_position[1]/region_dims[1])*dims[0] +
						(neighbor_position[2]/region_dims[2])*dims[0]*dims[1];

					// same simulation of simplicity as in the merge forest
					if (data[global_idx] == data[neighbor_global_idx]) {
						if (region_idx == neighbor_region_idx) {
							all_lower &= local_idx > neighbor_local_idx;
						} else {
							all_lower &= region_idx > neighbor_region_idx;
						}
					} else {
						all_lower &= data[global_idx] > data[neighbor_global_idx];
					}
				}

				if (all_lower) {
					if (maximum_count == maximum_capacity) {
						maximum_capacity *= 2;
						int64_t *tmp = realloc(maxima, maximum_capacity*sizeof *tmp);
						assert(tmp != NULL);
						maxima = tmp;
					}

					maxima[maximum_count++] = global_idx;
				}

				global_idx++;
			}
		}
	}

	*out_maxima = maxima;
	*out_maximum_count = maximum_count;
}