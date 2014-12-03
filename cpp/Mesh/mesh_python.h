////////////////////////////////////////////
// File: mesh_python.h                    //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
#ifndef MESH_MESH_PYTHON_H
#define MESH_MESH_PYTHON_H

// Includes
#include <algorithm>
#include <iterator>
#include <vector>
#include "Math/linalg_python.h"

// mesh_python
namespace mesh_python {

using mesh::FaceArray;
using mesh::GeneralMesh;

// PyArrayObject_to_CellArray
std::vector<int> PyArrayObject_to_CellArray(
    PyArrayObject * npy_raw_face_array) {
  // Construct `raw_face_array` from `npy_raw_face_array`.
  assert(PyArray_NDIM(npy_raw_face_array) == 1);
  auto v = linalg_python::PyArrayObject_to_VectorMap<int>(
    npy_raw_face_array);

  std::vector<int> raw_face_array;
  std::copy(v->data(), v->data() + v->size(), back_inserter(raw_face_array));
  return raw_face_array;
}

} // namespace mesh_python

#endif // MESH_MESH_PYTHON_H
