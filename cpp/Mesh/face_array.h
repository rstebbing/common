// face_array.h
#ifndef MESH_FACE_ARRAY_H
#define MESH_FACE_ARRAY_H

// Includes
#include <algorithm>
#include <cassert>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

// mesh
namespace mesh {

// FaceArray
class FaceArray {
 public:
  explicit FaceArray(const std::vector<int>& cell_array)
      : cell_array_(cell_array) {
    SetFaceOffsets();
  }

  explicit FaceArray(std::vector<int>&& cell_array)
      : cell_array_(std::move(cell_array)) {
    SetFaceOffsets();
  }

  inline const std::vector<int>& cell_array() const {
    return cell_array_;
  }

  inline int number_of_faces() const {
    return n_;
  }

  inline const int* face(int face_index) const {
    assert(face_index < number_of_faces());
    return &cell_array_[face_offsets_[face_index] + 1];
  }

  inline int number_of_sides(int face_index) const {
    assert(face_index < number_of_faces());
    return cell_array_[face_offsets_[face_index]];
  }

  std::vector<int> Faces(const std::vector<int>& face_indices) const {
    std::vector<int> cell_array;
    cell_array.push_back(static_cast<int>(face_indices.size()));

    for (int face_index : face_indices) {
      int n = number_of_sides(face_index);
      cell_array.push_back(n);

      auto f = face(face_index);
      std::copy(f, f + n, std::back_inserter(cell_array));
    }

    return cell_array;
  }

  void EnsureVertices() {
    if (vertices_.size() > 0) {
      return;
    }

    for (int i = 0; i < number_of_faces(); ++i) {
      auto f = face(i);
      std::copy(f, f + number_of_sides(i), std::back_inserter(vertices_));
    }

    std::sort(vertices_.begin(), vertices_.end());
    vertices_.resize(std::distance(vertices_.begin(),
                                   std::unique(vertices_.begin(),
                                               vertices_.end())));
  }

  inline const std::vector<int>& vertices() const {
    return vertices_;
  }

  inline int number_of_vertices() const {
    return static_cast<int>(vertices_.size());
  }

  void EnsureHalfEdgeToFaceIndex() {
    if (half_edge_to_face_index_.size() > 0) {
      return;
    }

    for (int i = 0; i < number_of_faces(); ++i) {
      auto f = face(i);
      int n = number_of_sides(i);
      for (int j = 0; j < n; ++j) {
        auto half_edge = std::make_pair(f[j], f[(j + 1) % n]);
        half_edge_to_face_index_[half_edge] = i;
      }
    }
  }

  int HalfEdgeToFaceIndex(int i, int j) const {
    auto k = half_edge_to_face_index_.find(std::make_pair(i, j));
    if (k != half_edge_to_face_index_.end()) {
      return k->second;
    } else {
      return -1;
    }
  }

  int FindCommonVertex() const {
    std::map<int, int> vertex_counts;

    for (int i = 0; i < number_of_faces(); ++i) {
      auto f = face(i);
      for (int j = 0; j < number_of_sides(i); ++j) {
        vertex_counts[f[j]] += 1;
      }
    }

    for (auto& i : vertex_counts) {
      if (i.second == number_of_faces()) {
        return i.first;
      }
    }

    return -1;
  }

  void RotateFaceToVertex(int face_index, const int i) {
    int* p = &cell_array_[face_offsets_[face_index]];
    int n = *p;
    ++p;

    int* q = std::find(p, p + n, i);
    assert(q != p + n);
    std::rotate(p, q, p + n);
  }

  void PermuteFaces(const std::vector<int>& permutation) {
    assert(permutation.size() == face_offsets_.size());

    std::vector<int> permuted_cell_array;
    permuted_cell_array.resize(cell_array_.size());
    permuted_cell_array[0] = number_of_faces();

    int j = 1;
    for (int i = 0; i < number_of_faces(); ++i) {
      assert(permutation[i] < face_offsets_.size());

      auto face_offset = face_offsets_[permutation[i]];
      auto p = &cell_array_[face_offset];
      auto n = *p;
      std::copy(p, p + n + 1, &permuted_cell_array[j]);
      j += n + 1;
    }

    cell_array_ = std::move(permuted_cell_array);
    SetFaceOffsets();
  }

 private:
  void SetFaceOffsets() {
    n_ = cell_array_[0];
    face_offsets_.resize(n_);

    int j = 1;
    for (int i = 0; i < n_; ++i) {
      face_offsets_[i] = j;
      j += cell_array_[j] + 1;
    }
  }

 private:
  std::vector<int> cell_array_;
  std::vector<int> face_offsets_;
  int n_;

  std::vector<int> vertices_;

  std::map<std::pair<int, int>, int> half_edge_to_face_index_;
};

} // namespace mesh

#endif // MESH_FACE_ARRAY_H
