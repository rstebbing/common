// linalg_python.h
#ifndef MATH_LINALG_PYTHON_H
#define MATH_LINALG_PYTHON_H

// Includes
#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include "linalg.h"
#include "linalg_typecodes.h"

// linalg_python
namespace linalg_python {

// PyArrayObject_to_MatrixMap
template <typename TElem>
std::unique_ptr<typename linalg::MatrixMap<TElem>::Type>
PyArrayObject_to_MatrixMap(PyArrayObject * npy_A)
{
  assert(PyArray_NDIM(npy_A) == 2);
  assert(PyArray_ISFORTRAN(npy_A));
  typedef typename linalg::MatrixMap<TElem>::Type MatrixMap;

  return std::unique_ptr<MatrixMap>(new MatrixMap(
    reinterpret_cast<TElem *>(PyArray_DATA(npy_A)),
                              PyArray_DIM(npy_A, 0),
                              PyArray_DIM(npy_A, 1)));
}

// PyArrayObject_to_VectorMap
template <typename TElem>
std::unique_ptr<typename linalg::VectorMap<TElem>::Type>
PyArrayObject_to_VectorMap(PyArrayObject * npy_A)
{
  assert(PyArray_NDIM(npy_A) == 1);
  assert(PyArray_ISCONTIGUOUS(npy_A));
  typedef typename linalg::VectorMap<TElem>::Type VectorMap;

  return std::unique_ptr<VectorMap>(new VectorMap(
    reinterpret_cast<TElem *>(PyArray_DATA(npy_A)),
                              PyArray_DIM(npy_A, 0)));
}

// PyList_to_VectorOfMatrixMap
template <typename TElem>
std::vector<std::unique_ptr<typename linalg::MatrixMap<TElem>::Type>>
PyList_to_VectorOfMatrixMap(PyObject * py_L)
{
  typedef typename linalg::MatrixMap<TElem>::Type MatrixMap;
  std::vector<std::unique_ptr<MatrixMap>> v;
  Py_ssize_t l = PyList_GET_SIZE(py_L);

  for (Py_ssize_t i = 0; i < l; ++i)
  {
    PyArrayObject * npy_A = (PyArrayObject *)PyList_GET_ITEM(py_L, i);

    assert(npy_A != nullptr);
    v.push_back(PyArrayObject_to_MatrixMap<TElem>(npy_A));
  }

  return v;
}

// PyList_to_VectorOfVectorMap
template <typename TElem>
std::vector<std::unique_ptr<typename linalg::VectorMap<TElem>::Type>>
PyList_to_VectorOfVectorMap(PyObject * py_L)
{
  typedef typename linalg::VectorMap<TElem>::Type VectorMap;
  std::vector<std::unique_ptr<VectorMap>> v;
  Py_ssize_t l = PyList_GET_SIZE(py_L);

  for (Py_ssize_t i = 0; i < l; ++i)
  {
    PyArrayObject * npy_A = (PyArrayObject *)PyList_GET_ITEM(py_L, i);

    assert(npy_A != nullptr);
    v.push_back(PyArrayObject_to_VectorMap<TElem>(npy_A));
  }

  return v;
}

// ScipyCSRMatrix_to_CSRMatrixMap
template <typename TElem>
std::unique_ptr<typename linalg::CSRMatrixMap<TElem>::Type>
ScipyCSRMatrix_to_CSRMatrixMap(PyObject * py_A)
{
  typedef typename linalg::CSRMatrixMap<TElem>::Type CSRMatrixMap;
  PyArrayObject * npy_indptr = (PyArrayObject *)PyObject_GetAttrString(
    py_A, "indptr");
  assert(npy_indptr != nullptr);
  auto indptr = PyArrayObject_to_VectorMap<int>(npy_indptr);

  PyArrayObject * npy_indices = (PyArrayObject *)PyObject_GetAttrString(
    py_A, "indices");
  assert(npy_indices != nullptr);
  auto indices = PyArrayObject_to_VectorMap<int>(npy_indices);

  PyArrayObject * npy_data = (PyArrayObject *)PyObject_GetAttrString(
    py_A, "data");
  assert(npy_data != nullptr);
  auto data = PyArrayObject_to_VectorMap<TElem>(npy_data);

  PyObject * py_shape = PyObject_GetAttrString(py_A, "shape");
  assert(py_shape != nullptr);

  PyObject * py_rows = PyTuple_GET_ITEM(py_shape, 0);
  assert(py_rows != nullptr);
  int rows = PyInt_AS_LONG(py_rows);

  PyObject * py_cols = PyTuple_GET_ITEM(py_shape, 1);
  assert(py_cols != nullptr);
  int cols = PyInt_AS_LONG(py_cols);

  std::unique_ptr<CSRMatrixMap> A(new CSRMatrixMap(
    rows, cols, static_cast<int>(indices->size()),
    indptr->data(), indices->data(), data->data()));

  Py_DECREF(py_shape);
  Py_DECREF((PyObject *)npy_data);
  Py_DECREF((PyObject *)npy_indices);
  Py_DECREF((PyObject *)npy_indptr);

  return A;
}

} // namespace linalg_python

#endif // MATH_LINALG_PYTHON_H
