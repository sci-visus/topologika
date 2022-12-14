// public domain license

// ASSUMPTIONS:
//	- persistence/triplet query would be run only for few maxima, making the Python overhead negligible

// LIMITATIONS:
//	- 3D numpy arrays
//	- power of two regions
//	- superlevel-set analysis

// TODO(11/29/2021): Measure that we are correct with respect to API and performance does not deteriorate.
// TODO(9/11/2021): should we call queries just 'maxima' instead of 'query_maxima'
// TODO(11/27/2019): the maxima query returns empty list if there are no maxima, but
//	the component query returns None if the component  does not exist; is it going to confuse
//	a user?

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_3_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>

#define TOPOLOGIKA_MERGE_FOREST_IMPLEMENTATION
#include "topologika_merge_forest.h"


typedef struct {
	PyObject_HEAD
	struct topologika_domain *domain;
	struct topologika_merge_forest *forest;
	int64_t dims[3];
} TopologikaMergeForestObject;



static void
TopologikaMergeForest_dealloc(TopologikaMergeForestObject *self)
{
	if (self->domain == NULL && self->forest == NULL) {
		Py_TYPE(self)->tp_free((PyObject *)self);
		return;
	}

	for (int64_t i = 0; i < self->forest->merge_tree_count; i++) {
		free(self->domain->regions[i].data);

		free(self->forest->merge_trees[i].arcs);
		free(self->forest->merge_trees[i].segmentation_offsets);
		free(self->forest->merge_trees[i].segmentation_counts);
		free(self->forest->merge_trees[i].segmentation);
		free(self->forest->merge_trees[i].vertex_to_arc);
		free(self->forest->merge_trees[i].reduced_bridge_set);
		free(self->forest->merge_trees[i].reduced_bridge_set_offsets);
		free(self->forest->merge_trees[i].reduced_bridge_set_counts);
	}
	free(self->domain);
	free(self->forest);
	Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject*
TopologikaMergeForest_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	TopologikaMergeForestObject *self = (TopologikaMergeForestObject *)type->tp_alloc(type, 0);
	if (self == NULL) {
		return (PyObject *)self;
	}

	self->domain = NULL;
	self->forest = NULL;

	return (PyObject *)self;
}


// TODO(3/19/2020): limit the region_dims to power of 2? (then the library can use shifts always for all regions except for incomplete ones if data dimensions are not divisible by region size)
// TODO(3/19/2020): double check that early returns free memory and decrement reference counts if needed
static int
TopologikaMergeForest_init(TopologikaMergeForestObject *self, PyObject *args, PyObject *kwds)
{
	char *kwlist[] = {"array", "region_shape", NULL};
	PyArrayObject *array = NULL;
	PyObject *region_dims_list = NULL;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", kwlist, &PyArray_Type, &array, &region_dims_list)) {
		return -1;
	}

	if (PyArray_TYPE(array) != NPY_FLOAT32) {
		PyErr_SetString(PyExc_ValueError, "The input NumPy array needs to have float32 type. An array can be converted by using array.astype(np.float32).");
		return -1;
	}
	if (PyArray_NDIM(array) != 3) {
		PyErr_Format(PyExc_ValueError, "The input NumPy array needs to be 3 dimensional. The passed array has %d dimensions.", PyArray_NDIM(array));
		return -1;
	}
	if (!PyArray_CHKFLAGS(array, NPY_ARRAY_C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_ValueError, "The input NumPy array needs to be C contiguous.");
		return -1;
	}
	if (region_dims_list != NULL && PyList_Size(region_dims_list) != PyArray_NDIM(array)) {
		PyErr_Format(PyExc_ValueError, "Region dimensions (%d) do not match array dimensions (%d).", PyList_Size(region_dims_list), PyArray_NDIM(array));
		return -1;
	}

	int64_t region_shape[3] = {64, 64, 64};
	if (region_dims_list != NULL) {
		for (int64_t i = 0; i < 3; i++) {
			if (i < PyList_Size(region_dims_list)) {
				PyObject *obj = PyList_GetItem(region_dims_list, i);
				if (!PyLong_Check(obj)) {
					PyErr_SetString(PyExc_TypeError, "Region shape needs to be integers");
					return -1;
				}
				region_shape[i] = PyLong_AsLongLong(obj);
				if (PyErr_Occurred() != NULL) {
					PyErr_SetString(PyExc_ValueError, "TODO");
					return -1;
				}
			} else {
				region_shape[i] = 1;
			}
		}
	}

	if (region_shape[0] < 1 || region_shape[1] < 1 || region_shape[2] < 1) {
		PyErr_SetString(PyExc_ValueError, "Region shape must be positive");
		return -1;
	}

	int64_t region_dims[] = {region_shape[2], region_shape[1], region_shape[0]}; // TODO: use region_shape in topologika too

	if (region_dims[0]*region_dims[1]*region_dims[2] >= TOPOLOGIKA_LOCAL_MAX) {
		assert(1024*1024*1024 < TOPOLOGIKA_LOCAL_MAX);
		PyErr_SetString(PyExc_ValueError, "Region shape is larger than the local index can represent. Maximum region shape is (1024, 1024, 1024).");
		return -1;
	}

	// we use dims[0] as width, dims[1] as height, and dims[2] as depth (which is reverse order
	//	than that of a numpy array
	// TODO(2/27/2020): should we switch to numpy's way of representing the dimensions of a 3D matrix?
	if (PyArray_NDIM(array) == 1) {
		self->dims[0] = PyArray_DIM(array, 0);
		self->dims[1] = 1;
		self->dims[2] = 1;
	} else if (PyArray_NDIM(array) == 2) {
		self->dims[0] = PyArray_DIM(array, 1);
		self->dims[1] = PyArray_DIM(array, 0);
		self->dims[2] = 1;
	} else {
		self->dims[0] = PyArray_DIM(array, 2);
		self->dims[1] = PyArray_DIM(array, 1);
		self->dims[2] = PyArray_DIM(array, 0);
	}

	switch (PyArray_TYPE(array)) {
	case NPY_FLOAT: {
		double construction_time_sec = 0.0;
		enum topologika_result result = topologika_compute_merge_forest_from_grid(PyArray_DATA(array), self->dims, region_dims, &self->domain, &self->forest, &construction_time_sec);
		if (result == topologika_error_out_of_memory) {
			PyErr_NoMemory(); // TODO: set estimate of needed memory
			return -1;
		}
		break;
	}
	default:
		return -1;
	}

	return 0;
}

static PyTypeObject TopologikaMergeForestType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "topologika.MergeForest",
	.tp_doc = "Merge forest that uses 3D coordinate conversions",
	.tp_basicsize = sizeof (TopologikaMergeForestObject),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = TopologikaMergeForest_new,
	.tp_init = (initproc)TopologikaMergeForest_init,
	.tp_dealloc = (destructor)TopologikaMergeForest_dealloc,
};




static PyObject *
query_maxima(PyObject  *self, PyObject *arg)
{
	if (PyObject_TypeCheck(arg, &TopologikaMergeForestType)) {
		TopologikaMergeForestObject *object = (TopologikaMergeForestObject *)arg;

		struct topologika_vertex *maxima = NULL;
		int64_t maximum_count = 0;

		// TODO: use scratch buffer
		enum topologika_result result = topologika_query_maxima(object->domain, object->forest, &maxima, &maximum_count);
		if (result == topologika_error_out_of_memory) {
			return PyErr_Format(PyExc_MemoryError, "Needed %"PRIi64" bytes of memory to complete maxima query.", 0); // TODO
		}

		PyObject *maxima_list = PyList_New(maximum_count);
		for (int64_t i = 0; i < maximum_count; i += 1) {
			int64_t coordinates[3];
			topologika_vertices_to_global_coordinates(object->dims, object->domain, &maxima[i], 1, &coordinates[0], &coordinates[1], &coordinates[2]);

			PyObject *tuple = PyTuple_New(3);
			PyList_SetItem(maxima_list, i, tuple);
			PyTuple_SetItem(tuple, 0, PyLong_FromLongLong(coordinates[2]));
			PyTuple_SetItem(tuple, 1, PyLong_FromLongLong(coordinates[1]));
			PyTuple_SetItem(tuple, 2, PyLong_FromLongLong(coordinates[0]));
		}

		free(maxima);

		return maxima_list;
	}

	return PyErr_Format(PyExc_TypeError, "Expect MergeForest type or ndarray.");
}




static PyObject *
query_componentmax(TopologikaMergeForestObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"", "vertex", "threshold", NULL};
	PyObject *arg = NULL;
	int64_t coordinates[3] = {0};
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O(LLL)d", kwlist, &arg, &coordinates[2], &coordinates[1], &coordinates[0], &threshold)) {
		return NULL;
	}

	if (PyObject_TypeCheck(arg, &TopologikaMergeForestType)) {
		TopologikaMergeForestObject *object = (TopologikaMergeForestObject *)arg;

		// TODO: probably forest query itself should do this check
		if (coordinates[0] < 0 || coordinates[0] >= object->dims[0] || coordinates[1] < 0 || coordinates[1] >= object->dims[1] || coordinates[2] < 0 || coordinates[2] >= object->dims[2]) {
			return PyErr_Format(PyExc_ValueError, "The vertex (%"PRIi64",%"PRIi64",%"PRIi64") lies outside the domain (%"PRIi64",%"PRIi64",%"PRIi64")",
				coordinates[2], coordinates[1], coordinates[0], object->dims[2], object->dims[1], object->dims[0]);
		}

		int64_t global_vertex_index = coordinates[0] + coordinates[1]*object->dims[0] + coordinates[2]*object->dims[0]*object->dims[1];

		// convert global vertex index to a pair of region index and local vertex index
		// TODO: simplify; technically, lldiv could be used, but it seems only MSVC compiles it down to idiv
		struct topologika_vertex vertex = topologika_global_index_to_vertex(object->dims, object->domain, global_vertex_index);
		struct topologika_vertex max_vertex = {0};
		enum topologika_result result = topologika_query_componentmax(object->domain, object->forest, vertex, threshold, &max_vertex);
		if (result == topologika_error_out_of_memory) {
			return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete the component max query.");
		}
		if (max_vertex.region_index == TOPOLOGIKA_LOCAL_MAX && max_vertex.vertex_index == TOPOLOGIKA_LOCAL_MAX) {
			Py_RETURN_NONE;
		}

		// convert the pair to global vertex index
		int64_t global_max_vertex_index = topologika_vertex_to_global_index(object->dims, object->domain, max_vertex);
		
		int64_t max_coordinates[3] = {
			global_max_vertex_index%object->dims[0],
			global_max_vertex_index/object->dims[0]%object->dims[1],
			global_max_vertex_index/object->dims[0]/object->dims[1],
		};
		return PyTuple_Pack(3, PyLong_FromLongLong(max_coordinates[2]), PyLong_FromLongLong(max_coordinates[1]), PyLong_FromLongLong(max_coordinates[0]));
	} else if (PyObject_TypeCheck(arg, &PyArray_Type)) {

	}

	return PyErr_Format(PyExc_TypeError, "Expect MergeForest or ndarray");
}



static PyObject *
query_component(PyObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"", "vertex", "threshold", NULL};
	PyObject *arg = NULL;
	int64_t coordinates[3] = {0};
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O(LLL)d", kwlist, &arg, &coordinates[2], &coordinates[1], &coordinates[0], &threshold)) {
		return NULL;
	}

	if (PyObject_TypeCheck(arg, &TopologikaMergeForestType)) {
		TopologikaMergeForestObject *object = (TopologikaMergeForestObject *)arg;

		// TODO: probably forest query itself should do this check
		if (coordinates[0] < 0 || coordinates[0] >= object->dims[0] || coordinates[1] < 0 || coordinates[1] >= object->dims[1] || coordinates[2] < 0 || coordinates[2] >= object->dims[2]) {
			return PyErr_Format(PyExc_ValueError, "The vertex (%"PRIi64",%"PRIi64",%"PRIi64") lies outside the domain (%"PRIi64",%"PRIi64",%"PRIi64")",
				coordinates[2], coordinates[1], coordinates[0], object->dims[2], object->dims[1], object->dims[0]);
		}

		int64_t global_vertex_index = coordinates[0] + coordinates[1]*object->dims[0] + coordinates[2]*object->dims[0]*object->dims[1];

		// convert global vertex index to a pair of region index and local vertex index
		// TODO: simplify; technically, lldiv could be used, but it seems only MSVC compiles it down to idiv
		struct topologika_vertex vertex = topologika_global_index_to_vertex(object->dims, object->domain, global_vertex_index);

		struct topologika_component *component = NULL;
		enum topologika_result result = topologika_query_component(object->domain, object->forest, vertex, threshold, &component);
		if (result == topologika_error_out_of_memory) {
			return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete component query.");
		}
		// TODO(11/27/2019): output an empty array for component query that has no solution (bottom)?
		if (result == topologika_error_no_output) {
			Py_RETURN_NONE;
		}

		PyObject *tuple = PyTuple_New(3);

		npy_intp dims[] = {(npy_intp)component->count};
		PyObject *xs = PyArray_SimpleNew(1, dims, NPY_INT16);
		PyObject *ys = PyArray_SimpleNew(1, dims, NPY_INT16);
		PyObject *zs = PyArray_SimpleNew(1, dims, NPY_INT16);

		PyTuple_SetItem(tuple, 0, zs);
		PyTuple_SetItem(tuple, 1, ys);
		PyTuple_SetItem(tuple, 2, xs);

		topologika_vertices_to_global_coordinates16(object->dims, object->domain, component->data, component->count, (int16_t *)PyArray_GETPTR1(xs, 0), (int16_t *)PyArray_GETPTR1(ys, 0), (int16_t *)PyArray_GETPTR1(zs, 0));

		free(component);

		return tuple;
	}

	return PyErr_Format(PyExc_TypeError, "Expect MergeForest or ndarray");
}



// TODO(9/2/2021): extract segmentation lazily when it is accessed?
static PyObject *
query_components(PyObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"", "threshold", NULL};
	PyObject *o = NULL;
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Od", kwlist, &o, &threshold)) {
		return NULL;
	}

	if (PyObject_TypeCheck(o, &TopologikaMergeForestType)) {
		TopologikaMergeForestObject *object = (TopologikaMergeForestObject *)o;

		int64_t start = topologika_usec_counter();
		struct topologika_component **components = NULL;
		int64_t component_count = 0;
		enum topologika_result result = topologika_query_components(object->domain, object->forest, threshold,
			&components, &component_count);
		if (result == topologika_error_out_of_memory) {
			// TODO: which one if we run many of them?
			return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete components query.");
		}
		int64_t end = topologika_usec_counter();
		//printf("SERIAL EXTRACT TOOK %f s\n", (end - start)*1e-6);

		// convert to global coordinates and Python list
		start = topologika_usec_counter();
		PyObject *components_list = PyList_New(component_count);
		for (int64_t i = 0; i < component_count; i++) {
			struct topologika_component *component = components[i];

			npy_intp dims[] = {(npy_intp)component->count};
			PyObject *list = PyArray_SimpleNew(1, dims, NPY_INT64);
			PyList_SetItem(components_list, i, list);

			topologika_vertices_to_global_indices(object->dims, object->domain, component->data, component->count, (int64_t *)PyArray_GETPTR1(list, 0));

			free(component);
		}
		end = topologika_usec_counter();
		//printf("CONVERSION TOOK %f s\n", (end - start)*1e-6);

		free(components);

		return components_list;
	} else if (PyObject_TypeCheck(o, &TopologikaMergeForestType)) {
		TopologikaMergeForestObject *object = (TopologikaMergeForestObject *)o;

		int64_t start = topologika_usec_counter();
		struct topologika_component **components = NULL;
		int64_t component_count = 0;
		enum topologika_result result = topologika_query_components(object->domain, object->forest, threshold,
			&components, &component_count);
		if (result == topologika_error_out_of_memory) {
			// TODO: which one if we run many of them?
			return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete components query.");
		}
		int64_t end = topologika_usec_counter();
		//printf("1: SERIAL EXTRACT TOOK %f s\n", (end - start)*1e-6);

		// convert to global coordinates and Python list
		start = topologika_usec_counter();
		PyObject *components_list = PyList_New(component_count);
		for (int64_t i = 0; i < component_count; i++) {
			struct topologika_component *component = components[i];

			PyObject *tuple = PyTuple_New(3);
			PyList_SetItem(components_list, i, tuple);

			npy_intp dims[] = {(npy_intp)component->count};
			PyObject *xs = PyArray_SimpleNew(1, dims, NPY_INT16);
			PyObject *ys = PyArray_SimpleNew(1, dims, NPY_INT16);
			PyObject *zs = PyArray_SimpleNew(1, dims, NPY_INT16);

			PyTuple_SetItem(tuple, 0, zs);
			PyTuple_SetItem(tuple, 1, ys);
			PyTuple_SetItem(tuple, 2, xs);

			topologika_vertices_to_global_coordinates16(object->dims, object->domain, component->data, component->count, (int16_t *)PyArray_GETPTR1(xs, 0), (int16_t *)PyArray_GETPTR1(ys, 0), (int16_t *)PyArray_GETPTR1(zs, 0));

			free(component);
		}
		end = topologika_usec_counter();
		//printf("1: CONVERSION TOOK %f s\n", (end - start)*1e-6);

		free(components);

		return components_list;
	}

	return PyErr_Format(PyExc_TypeError, "Expect MergeForest or MergeForest1");
}


static PyObject *
query_persistence(PyObject* self, PyObject* args, PyObject* keywds)
{

	char *kwlist[] = {"", "vertex", NULL};
	TopologikaMergeForestObject *object = NULL;
	int64_t coordinates[3] = {0};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!(LLL)", kwlist, &TopologikaMergeForestType, &object, &coordinates[2], &coordinates[1], &coordinates[0])) {
		return NULL;
	}

	// TODO: probably forest query itself should do this check
	if (coordinates[0] < 0 || coordinates[0] >= object->dims[0] || coordinates[1] < 0 || coordinates[1] >= object->dims[1] || coordinates[2] < 0 || coordinates[2] >= object->dims[2]) {
		return PyErr_Format(PyExc_ValueError, "The vertex (%"PRIi64",%"PRIi64",%"PRIi64") lies outside the domain (%"PRIi64",%"PRIi64",%"PRIi64")",
			coordinates[2], coordinates[1], coordinates[0], object->dims[2], object->dims[1], object->dims[0]);
	}

	int64_t global_vertex_index = coordinates[0] + coordinates[1]*object->dims[0] + coordinates[2]*object->dims[0]*object->dims[1];
	struct topologika_vertex localized_vertex = topologika_global_index_to_vertex(object->dims, object->domain, global_vertex_index);

	double persistence = 0.0;
	enum topologika_result result = topologika_query_persistencebelow(object->domain, object->forest, localized_vertex, INFINITY, &persistence);
	if (result != topologika_result_success) {
		// TODO
		return PyErr_Format(PyExc_BaseException, "Persistence query failed.");
	}

	return PyFloat_FromDouble(persistence);
}


static PyObject *
query_triplet(PyObject* self, PyObject* args, PyObject* keywds)
{
	char *kwlist[] = {"", "vertex", NULL};
	TopologikaMergeForestObject *object = NULL;
	int64_t coordinates[3] = {0};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!(LLL)", kwlist, &TopologikaMergeForestType, &object, &coordinates[2], &coordinates[1], &coordinates[0])) {
		return NULL;
	}

	// TODO: probably forest query itself should do this check
	if (coordinates[0] < 0 || coordinates[0] >= object->dims[0] || coordinates[1] < 0 || coordinates[1] >= object->dims[1] || coordinates[2] < 0 || coordinates[2] >= object->dims[2]) {
		return PyErr_Format(PyExc_ValueError, "The vertex (%"PRIi64",%"PRIi64",%"PRIi64") lies outside the domain (%"PRIi64",%"PRIi64",%"PRIi64")",
			coordinates[2], coordinates[1], coordinates[0], object->dims[2], object->dims[1], object->dims[0]);
	}

	int64_t global_vertex_index = coordinates[0] + coordinates[1]*object->dims[0] + coordinates[2]*object->dims[0]*object->dims[1];
	struct topologika_vertex localized_vertex = topologika_global_index_to_vertex(object->dims, object->domain, global_vertex_index);

	struct topologika_triplet triplet;
	enum topologika_result result = topologika_query_triplet(object->domain, object->forest, localized_vertex, &triplet);
	if (result != topologika_result_success) {
		// TODO
		return PyErr_Format(PyExc_BaseException, "Triplet query failed.");
	}

	if (topologika_vertex_eq(triplet.u, triplet.s) && topologika_vertex_eq(triplet.u, triplet.v)) {
		Py_RETURN_NONE;
	}

	int64_t global_u_vertex_index = topologika_vertex_to_global_index(object->dims, object->domain, triplet.u);
	int64_t u_coordinates[3] = {
		global_u_vertex_index%object->dims[0],
		global_u_vertex_index/object->dims[0]%object->dims[1],
		global_u_vertex_index/object->dims[0]/object->dims[1],
	};

	int64_t global_s_vertex_index = topologika_vertex_to_global_index(object->dims, object->domain, triplet.s);
	int64_t s_coordinates[3] = {
		global_s_vertex_index%object->dims[0],
		global_s_vertex_index/object->dims[0]%object->dims[1],
		global_s_vertex_index/object->dims[0]/object->dims[1],
	};

	int64_t global_v_vertex_index = topologika_vertex_to_global_index(object->dims, object->domain, triplet.v);
	int64_t v_coordinates[3] = {
		global_v_vertex_index%object->dims[0],
		global_v_vertex_index/object->dims[0]%object->dims[1],
		global_v_vertex_index/object->dims[0]/object->dims[1],
	};

	return PyTuple_Pack(3,
		PyTuple_Pack(3, PyLong_FromLongLong(u_coordinates[2]), PyLong_FromLongLong(u_coordinates[1]), PyLong_FromLongLong(u_coordinates[0])),
		PyTuple_Pack(3, PyLong_FromLongLong(s_coordinates[2]), PyLong_FromLongLong(s_coordinates[1]), PyLong_FromLongLong(s_coordinates[0])),
		PyTuple_Pack(3, PyLong_FromLongLong(v_coordinates[2]), PyLong_FromLongLong(v_coordinates[1]), PyLong_FromLongLong(v_coordinates[0])));
}






static PyMethodDef methods[] = {
	{"maxima", (PyCFunction)query_maxima, METH_O, "Returns all maxima in the dataset."},
	{"componentmax", (PyCFunction)query_componentmax, METH_VARARGS | METH_KEYWORDS, "Return maximum in a component given maximum and threshold."},
	{"component", (PyCFunction)query_component, METH_VARARGS | METH_KEYWORDS, "Return component given maximum and threshold."},
	{"components", (PyCFunction)query_components, METH_VARARGS | METH_KEYWORDS, "Return vertices of all connected components at the given threshold."},
	{"persistence", (PyCFunction)query_persistence, METH_VARARGS | METH_KEYWORDS, "Returns persistence of a given maximum."},
	{"triplet", (PyCFunction)query_triplet, METH_VARARGS | METH_KEYWORDS, "Returns triplet of a given maximum."},
	{NULL},
};


static PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	.m_name = "topologika",
	.m_doc = "Topological queries based on forest data structure.",
	.m_size = -1,
	.m_methods = methods,
};


PyMODINIT_FUNC
PyInit_topologika(void)
{
	// TODO: static asserts
	assert(sizeof (long long) == sizeof (int64_t));

	// TODO(2/28/2020): segfault in PyType_Ready if the library is compiled in a debug mode
	//	python setup.py build -g -f && python setup.py install --user
	//	python -m unittest test_persistence.py
	if (PyType_Ready(&TopologikaMergeForestType) < 0) {
		return NULL;
	}

	PyObject *m = PyModule_Create(&module);
	if (m == NULL) {
		return NULL;
	}

	Py_INCREF(&TopologikaMergeForestType);
	if (PyModule_AddObject(m, "MergeForest", (PyObject *)&TopologikaMergeForestType) < 0) {
		Py_DECREF(&TopologikaMergeForestType);
		Py_DECREF(m);
		return NULL;
	}

	import_array();
	return m;
}
