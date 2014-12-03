////////////////////////////////////////////
// File: general_mesh.h                   //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
#ifndef MESH_GENERAL_MESH_H
#define MESH_GENERAL_MESH_H

// Includes
#include "mesh.h"

// mesh
namespace mesh {

// GeneralMeshBase
class GeneralMeshBase : public FaceArray {
 public:
  template <typename T>
  explicit GeneralMeshBase(T&& cell_array)
      : FaceArray(std::forward<T>(cell_array)) {
    Initialise();
  }

  inline int half_edge_face_index(int half_edge_index) const {
    return _half_edge_face_indices[half_edge_index];
  }

  inline int half_edge_face_offset(int half_edge_index) const {
    return _half_edge_offsets[half_edge_index];
  }

  inline int half_edge_index(int face_index, int edge_offset) const {
    return _face_offsets[face_index] + edge_offset;
  }

 private:
  void Initialise() {
    int k = 0;
    for (int i = 0; i < number_of_faces(); ++i) {
      _face_offsets.push_back(k);

      for (int j = 0; j < number_of_sides(i); ++j) {
        _half_edge_face_indices.push_back(i);
        _half_edge_offsets.push_back(j);
        ++k;
      }
    }
  }

 private:
  std::vector<int> _half_edge_face_indices;
  std::vector<int> _half_edge_offsets;
  std::vector<int> _face_offsets;
};

// GeneralMesh
typedef Mesh<GeneralMeshBase> GeneralMesh;

// begin (GeneralMesh::EdgeIterator)
inline GeneralMesh::EdgeIterator& begin(
    std::pair<GeneralMesh::EdgeIterator, GeneralMesh::EdgeIterator>& a) {
  return a.first;
}

// end (GeneralMesh::EdgeIterator)
inline GeneralMesh::EdgeIterator& end(
    std::pair<GeneralMesh::EdgeIterator, GeneralMesh::EdgeIterator>& a) {
  return a.second;
}

} // namespace mesh

#endif // MESH_GENERAL_MESH_H
