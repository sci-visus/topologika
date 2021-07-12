// public domain license

// TODO(11/27/2019): the maxima query returns empty list if there are no maxima, but
//	the component query returns None if the component  does not exist; is it going to confuse
//	a user?
// TODO(11/26/2019): support only numpy.block inputs? (currently we have to perform
//	conversions from region+local to global coordinates)
// TODO(11/12/2019): return results as numpy arrays?
// TODO(11/8/2019): raise idiomatic exeptions with good error messages (that also have examples of how to solve the problem)


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
	char *kwlist[] = {"array", "region_dims", NULL}; // TODO(3/19/2020): use numpy nomenclature to avoid confusion; also, we may need to use z,y,x order to match numpy's idioms
	PyArrayObject *array = NULL;
	PyObject *region_dims_list = NULL;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|O", kwlist, &PyArray_Type, &array, &region_dims_list)) {
		return -1;
	}
	if (array == NULL || !PyArray_CHKFLAGS(array, NPY_ARRAY_C_CONTIGUOUS) ||
		PyArray_TYPE(array) != NPY_FLOAT || PyArray_NDIM(array) != 3) {
		PyErr_SetString(PyExc_ValueError, "The input array must be 3D, currently it is TODO. For example, numpy.zeros((10, 10, 10)).");
		return -1;
	}
	if (region_dims_list != NULL && PyList_Size(region_dims_list) != PyArray_NDIM(array)) {
		PyErr_SetString(PyExc_ValueError, "Region dimensions do not match array dimensions."); // TODO(3/19/2020): better message
		return -1;
	}

	int64_t region_dims[3] = {64, 64, 64};
	if (region_dims_list != NULL) {
		for (int64_t i = 0; i < 3; i++) {
			if (i < PyList_Size(region_dims_list)) {
				PyObject *obj = PyList_GetItem(region_dims_list, i);
				region_dims[i] = PyLong_AsLongLong(obj);
				if (PyErr_Occurred() != NULL) {
					PyErr_SetString(PyExc_ValueError, "Region dimensions must be integers");
					return -1;
				}
			} else {
				region_dims[i] = 1;
			}
		}
	}

	// we use dims[0] as width, dims[1] as height, and dims[2] as depth (which is reverse order
	//	than that of a numpy array
	// TODO(2/27/2020): should we switch to numpy's way of representing the dimensions of a 3D matrix?
	self->dims[0] = PyArray_DIM(array, 2);
	self->dims[1] = PyArray_DIM(array, 1);
	self->dims[2] = PyArray_DIM(array, 0);

	switch (PyArray_TYPE(array)) {
	case NPY_FLOAT: {
		double construction_time_sec = 0.0;
		enum topologika_result result = topologika_compute_merge_forest_from_grid(PyArray_DATA(array), self->dims, region_dims, &self->domain, &self->forest, &construction_time_sec);
		if (result == topologika_error_out_of_memory) {
			PyErr_NoMemory(); // TODO: set estimate of needed memory
			// TODO: decrease ref count for numpy?
			return -1;
		}
		break;
	}
	default:
		return -1;
	}

	return 0;
}


// NOTE(11/12/2019): for legacy reasons, we take global index that we convert to region_idx + local_idx
//	and then we convert the resulting pair back to the global_idx
// TODO: should we initialize variables to -1 to cause crash if we forgot a check?
static PyObject *
TopologikaMergeForest_query_component_max(TopologikaMergeForestObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"vertex_index", "threshold", NULL};
	int64_t global_vertex_index = 0;
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Ld", kwlist, &global_vertex_index, &threshold)) {
		return NULL;
	}

	int64_t vertex_count = self->dims[0]*self->dims[1]*self->dims[2];
	if (global_vertex_index < 0 || global_vertex_index >= vertex_count) {
		return PyErr_Format(PyExc_ValueError, "The global index %"PRIi64" lies outside the domain (the range is [0, %"PRIi64"))", global_vertex_index, vertex_count);
	}

	// convert global vertex index to a pair of region index and local vertex index
	// TODO: simplify; technically, lldiv could be used, but it seems only MSVC compiles it down to idiv
	struct topologika_vertex vertex = topologika_global_index_to_vertex(self->dims, self->domain, global_vertex_index);

	struct topologika_vertex max_vertex = {0};
	enum topologika_result result = topologika_query_component_max(self->domain, self->forest, vertex, threshold, &max_vertex);
	if (result == topologika_error_out_of_memory) {
		return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete the component max query.");
	}
	if (result == topologika_error_no_output) {
		Py_RETURN_NONE;
	}

	// convert the pair to global vertex index
	int64_t global_max_vertex_index = topologika_vertex_to_global_index(self->dims, self->domain, max_vertex);
	return PyLong_FromLongLong(global_max_vertex_index);
}



// TODO(1/23/2020): converting to the global coordinate space may pose problem when we start relying on adaptive grid
// TODO(11/25/2019): return numpy array?
// TODO(11/12/2019): should the query take threshold? (discard maxima below it?)
static PyObject *
TopologikaMergeForest_query_maxima(TopologikaMergeForestObject *self, PyObject *args)
{
	struct topologika_vertex *maxima = NULL;
	int64_t maximum_count = 0;

	// TODO: use scratch buffer
	enum topologika_result result = topologika_query_maxima(self->domain, self->forest, &maxima, &maximum_count);
	if (result == topologika_error_out_of_memory) {
		// TODO: which one if we run many of them?
		return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete maxima query.");
	}

	// convert to global coordinates and Python list
	PyObject *list = PyList_New(maximum_count);
	for (int64_t i = 0; i < maximum_count; i++) {
		int64_t global_max_vertex_index = topologika_vertex_to_global_index(self->dims, self->domain, maxima[i]);
		PyList_SetItem(list, i, PyLong_FromLongLong(global_max_vertex_index));
	}

	free(maxima);

	return list;
}



// TODO(11/25/2019): return numpy array?
static PyObject *
TopologikaMergeForest_query_components(TopologikaMergeForestObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"threshold", NULL};
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "d", kwlist, &threshold)) {
		return NULL;
	}

	struct topologika_component **components = NULL;
	int64_t component_count = 0;
	enum topologika_result result = topologika_query_components(self->domain, self->forest, threshold,
		&components, &component_count);
	if (result == topologika_error_out_of_memory) {
		// TODO: which one if we run many of them?
		return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete components query.");
	}

	// convert to global coordinates and Python list
	PyObject *components_list = PyList_New(component_count);
	for (int64_t i = 0; i < component_count; i++) {
		struct topologika_component *component = components[i];

		PyObject *list = PyList_New(component->count);
		PyList_SetItem(components_list, i, list);
		for (int64_t j = 0; j < component->count; j++) {
			int64_t global_vertex_index = topologika_vertex_to_global_index(self->dims, self->domain, component->data[j]);
			PyList_SetItem(list, j, PyLong_FromLongLong(global_vertex_index));
		}

		free(component);
	}

	free(components);

	return components_list;
}




static PyObject *
TopologikaMergeForest_query_component(TopologikaMergeForestObject *self, PyObject *args, PyObject *keywds)
{
	char *kwlist[] = {"vertex_index", "threshold", NULL};
	int64_t global_vertex_index = 0;
	double threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Ld", kwlist, &global_vertex_index, &threshold)) {
		return NULL;
	}

	int64_t vertex_count = self->dims[0]*self->dims[1]*self->dims[2];
	if (global_vertex_index < 0 || global_vertex_index >= vertex_count) {
		return PyErr_Format(PyExc_ValueError, "The global index %"PRIi64" lies outside the domain (the range is [0, %"PRIi64"))", global_vertex_index, vertex_count);
	}

	struct topologika_vertex vertex = topologika_global_index_to_vertex(self->dims, self->domain, global_vertex_index);

	struct topologika_component *component = NULL;
	enum topologika_result result = topologika_query_component(self->domain, self->forest, vertex, threshold, &component);
	if (result == topologika_error_out_of_memory) {
		return PyErr_Format(PyExc_MemoryError, "Needed X GB of memory to complete component query.");
	}
	// TODO(11/27/2019): output an empty list for component query that has no solution (bottom)?
	if (result == topologika_error_no_output) {
		Py_RETURN_NONE;
	}

	// convert to global coordinates and Python list
	// TODO(2/8/2020): the Python's list has large overhead per element, use numpy array or generator
	PyObject *list = PyList_New(component->count);
	for (int64_t i = 0; i < component->count; i++) {
		int64_t global_vertex_index = topologika_vertex_to_global_index(self->dims, self->domain, component->data[i]);
		PyList_SetItem(list, i, PyLong_FromLongLong(global_vertex_index));
	}
	free(component);

	return list;
}




static PyObject *
TopologikaMergeForest_query_triplet(TopologikaMergeForestObject* self, PyObject* args, PyObject* keywds)
{
	char *kwlist[] = {"vertex_index", NULL};
	int64_t global_vertex_index = 0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "L", kwlist, &global_vertex_index)) {
		return NULL;
	}

	int64_t vertex_count = self->dims[0]*self->dims[1]*self->dims[2];
	if (global_vertex_index < 0 || global_vertex_index >= vertex_count) {
		return PyErr_Format(PyExc_ValueError, "The global index %"PRIi64" lies outside the domain (the range is [0, %"PRIi64"))", global_vertex_index, vertex_count);
	}

	struct topologika_vertex vertex = topologika_global_index_to_vertex(self->dims, self->domain, global_vertex_index);

	struct topologika_triplet triplet;
	enum topologika_result result = topologika_query_triplet(self->domain, self->forest, vertex, &triplet);
	if (result != topologika_result_success) {
		// TODO
		return PyErr_Format(PyExc_BaseException, "Triplet query failed.");
	}

	if (topologika_vertex_eq(triplet.u, triplet.s) && topologika_vertex_eq(triplet.u, triplet.v)) {
		Py_RETURN_NONE;
	}

	PyObject *tuple = PyTuple_New(3);
	PyTuple_SetItem(tuple, 0, PyLong_FromLongLong(topologika_vertex_to_global_index(self->dims, self->domain, triplet.u)));
	PyTuple_SetItem(tuple, 1, PyLong_FromLongLong(topologika_vertex_to_global_index(self->dims, self->domain, triplet.s)));
	PyTuple_SetItem(tuple, 2, PyLong_FromLongLong(topologika_vertex_to_global_index(self->dims, self->domain, triplet.v)));

	return tuple;
}




static PyObject *
TopologikaMergeForest_query_persistence(TopologikaMergeForestObject* self, PyObject* args, PyObject* keywds)
{
	char *kwlist[] = {"vertex_index", NULL};
	int64_t global_vertex_index = 0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "L", kwlist, &global_vertex_index)) {
		return NULL;
	}

	int64_t vertex_count = self->dims[0]*self->dims[1]*self->dims[2];
	if (global_vertex_index < 0 || global_vertex_index >= vertex_count) {
		return PyErr_Format(PyExc_ValueError, "The global index %"PRIi64" lies outside the domain (the range is [0, %"PRIi64"))", global_vertex_index, vertex_count);
	}

	struct topologika_vertex vertex = topologika_global_index_to_vertex(self->dims, self->domain, global_vertex_index);

	double persistence = 0.0f;
	enum topologika_result result = topologika_query_persistence(self->domain, self->forest, vertex, &persistence);
	if (result != topologika_result_success) {
		// TODO
		return PyErr_Format(PyExc_BaseException, "Persistence query failed.");
	}

	return PyFloat_FromDouble(persistence);
}




static PyObject *
TopologikaMergeForest_query_persistencebelow(TopologikaMergeForestObject* self, PyObject* args, PyObject* keywds)
{
	char *kwlist[] = {"vertex_index", "persistence_threshold", NULL};
	int64_t global_vertex_index = 0;
	double persistence_threshold = 0.0;
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Ld", kwlist, &global_vertex_index, &persistence_threshold)) {
		return NULL;
	}

	int64_t vertex_count = self->dims[0]*self->dims[1]*self->dims[2];
	if (global_vertex_index < 0 || global_vertex_index >= vertex_count) {
		return PyErr_Format(PyExc_ValueError, "The global index %"PRIi64" lies outside the domain (the range is [0, %"PRIi64"))", global_vertex_index, vertex_count);
	}

	struct topologika_vertex vertex = topologika_global_index_to_vertex(self->dims, self->domain, global_vertex_index);

	double persistence = 0.0f;
	enum topologika_result result = topologika_query_persistencebelow(self->domain, self->forest, vertex, persistence_threshold, &persistence);
	if (result != topologika_result_success) {
		// TODO
		return PyErr_Format(PyExc_BaseException, "Persistencebelow query failed.");
	}

	return PyFloat_FromDouble(persistence);
}




static PyMethodDef TopologikeMergeForest_methods[] = {
	{"query_component_max", (PyCFunction)TopologikaMergeForest_query_component_max, METH_VARARGS | METH_KEYWORDS, "Given a vertex and threshold return the highest vertex in a connected component containing the vertex at the specified threshold."},
	{"query_component", (PyCFunction)TopologikaMergeForest_query_component, METH_VARARGS | METH_KEYWORDS, "Given a vertex and threshold return the connected component containing the vertex."},
	{"query_components", (PyCFunction)TopologikaMergeForest_query_components, METH_VARARGS | METH_KEYWORDS, "Return vertices of all connected components at the given threshold."},
	{"query_maxima", (PyCFunction)TopologikaMergeForest_query_maxima, METH_NOARGS, "Returns all maxima in the data set."},
	{"query_triplet", (PyCFunction)TopologikaMergeForest_query_triplet, METH_VARARGS | METH_KEYWORDS, "Given a maximum return its triplet."},
	{"query_persistence", (PyCFunction)TopologikaMergeForest_query_persistence, METH_VARARGS | METH_KEYWORDS, "Given a maximum return its persistence."},
	{"query_persistencebelow", (PyCFunction)TopologikaMergeForest_query_persistencebelow, METH_VARARGS | METH_KEYWORDS, "Given a maximum and persistence threshold return its persistence if below the threshold, otherwise return infinity."},
	{NULL},
};


static PyTypeObject TopologikaMergeForestType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "topologika.MergeForest",
	.tp_doc = "Merge forest",
	.tp_basicsize = sizeof (TopologikaMergeForestObject),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT,
	.tp_new = TopologikaMergeForest_new,
	.tp_init = (initproc)TopologikaMergeForest_init,
	.tp_dealloc = (destructor)TopologikaMergeForest_dealloc,
	.tp_methods = TopologikeMergeForest_methods,
};


static PyModuleDef topologika_module = {
	PyModuleDef_HEAD_INIT,
	.m_name = "topologika",
	.m_doc = "Topological queries based on forest data structure.",
	.m_size = -1,
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

	PyObject* module = PyModule_Create(&topologika_module);
	if (module == NULL) {
		return NULL;
	}

	Py_INCREF(&TopologikaMergeForestType);
	if (PyModule_AddObject(module, "MergeForest", (PyObject*)&TopologikaMergeForestType) < 0) {
		Py_DECREF(&TopologikaMergeForestType);
		Py_DECREF(module);
		return NULL;
	}

	import_array();
	return module;
}
