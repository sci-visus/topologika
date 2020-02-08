#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_3_API_VERSION
#include <numpy/arrayobject.h>

#include <stdint.h>


typedef float data_t;
#include "topologika_merge_forest_reference.h"


typedef struct {
	PyObject_HEAD
	PyArrayObject* array;
	data_t *data;
	int64_t dims[3]; // TODO: duplicate with PyArray_DIM(array)
	int64_t region_dims[3];
} TopologikaReferenceMergeForestObject;




static void
TopologikaReferenceMergeForest_dealloc(TopologikaReferenceMergeForestObject *self)
{
	Py_DECREF(self->array);
	Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject*
TopologikaReferenceMergeForest_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	TopologikaReferenceMergeForestObject *self = (TopologikaReferenceMergeForestObject *)type->tp_alloc(type, 0);
	if (self == NULL) {
		return (PyObject *)self;
	}

	self->array = NULL;

	return (PyObject *)self;
}


static int
TopologikaReferenceMergeForest_init(TopologikaReferenceMergeForestObject *self, PyObject *args, PyObject *kwds)
{
	PyArrayObject* array = NULL;
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
		return -1;
	}
	if (array == NULL || !PyArray_CHKFLAGS(array, NPY_ARRAY_C_CONTIGUOUS) ||
		PyArray_TYPE(array) != NPY_FLOAT || PyArray_NDIM(array) != 3) {
		PyErr_SetString(PyExc_ValueError, "The input array must be 3D, currently it is BLA. For example, numpy.zeros((10, 10, 10)).");
		return -1;
	}

	Py_INCREF(array);
	self->array = array; // for reference counting to keep it alyive

	self->dims[0] = PyArray_DIM(array, 0);
	self->dims[1] = PyArray_DIM(array, 1);
	self->dims[2] = PyArray_DIM(array, 2);

	self->region_dims[0] = 64;
	self->region_dims[1] = 64;
	self->region_dims[2] = 64;

	switch (PyArray_TYPE(array)) {
	case NPY_FLOAT: {
		self->data = PyArray_DATA(array);
		break;
	}
	default:
		return -1;
	}

	return 0;
}


static PyObject *
TopologikaReferenceMergeForest_query_maxima(TopologikaReferenceMergeForestObject *self, PyObject *args)
{
	int64_t *maxima = NULL;
	int64_t maximum_count = 0;

	topologika_reference_query_maxima(self->data, self->dims, self->region_dims, &maxima, &maximum_count);

	// convert to global coordinates and Python list
	PyObject *list = PyList_New(maximum_count);
	for (int64_t i = 0; i < maximum_count; i++) {
		PyList_SetItem(list, i, PyLong_FromLongLong(maxima[i]));
	}

	free(maxima);

	return list;
}



static PyObject *
TopologikaReferenceMergeForest_query_component(TopologikaReferenceMergeForestObject *self, PyObject *args, PyObject *keywds)
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

	int64_t *component = NULL;
	int64_t component_count = 0;
	topologika_reference_query_component(self->data, self->dims, self->region_dims, global_vertex_index, threshold, &component, &component_count);

	if (component == NULL) {
		Py_RETURN_NONE;
	}

	// TODO(2/8/2020): the Python's list has large overhead per element, use numpy array
	PyObject *list = PyList_New(component_count);
	for (int64_t i = 0; i < component_count; i++) {
		PyList_SetItem(list, i, PyLong_FromLongLong(component[i]));
	}
	free(component);

	return list;
}





static PyMethodDef TopologikeReferenceMergeForest_methods[] = {
	{"query_component", (PyCFunction)TopologikaReferenceMergeForest_query_component, METH_VARARGS | METH_KEYWORDS, "Given a vertex and threshold return the connected component containing the vertex."},
	{"query_maxima", (PyCFunction)TopologikaReferenceMergeForest_query_maxima, METH_NOARGS, "Returns all maxima in the data set."},
	{NULL},
};


static PyTypeObject TopologikaReferenceMergeForestType = {
	PyVarObject_HEAD_INIT(NULL, 0)
	.tp_name = "topologika_reference.MergeForest",
	.tp_doc = "Merge forest",
	.tp_basicsize = sizeof(TopologikaReferenceMergeForestObject),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT, //| Py_TPFLAGS_BASETYPE,
	.tp_new = TopologikaReferenceMergeForest_new,
	.tp_init = (initproc)TopologikaReferenceMergeForest_init,
	.tp_dealloc = (destructor)TopologikaReferenceMergeForest_dealloc,
	.tp_methods = TopologikeReferenceMergeForest_methods,
};


static PyModuleDef topologikareference_module = {
	PyModuleDef_HEAD_INIT,
	.m_name = "topologika_reference",
	.m_doc = "Topological queries based on forest data structure.",
	.m_size = -1,
};


PyMODINIT_FUNC
PyInit_topologika_reference(void)
{
	// TODO: static asserts
	assert(sizeof (long long) == sizeof (int64_t));

	if (PyType_Ready(&TopologikaReferenceMergeForestType) < 0) {
		return NULL;
	}

	PyObject* module = PyModule_Create(&topologikareference_module);
	if (module == NULL) {
		return NULL;
	}

	Py_INCREF(&TopologikaReferenceMergeForestType);
	if (PyModule_AddObject(module, "MergeForest", (PyObject*)&TopologikaReferenceMergeForestType) < 0) {
		Py_DECREF(&TopologikaReferenceMergeForestType);
		Py_DECREF(module);
		return NULL;
	}

	import_array();
	return module;
}
