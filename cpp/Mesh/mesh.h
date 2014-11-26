// mesh.h
#ifndef MESH_MESH_H
#define MESH_MESH_H

// Includes
#include "face_array.h"

#include <queue>

// mesh
namespace mesh {

// Mesh
// From `MeshBase` (which inherits from `FaceArray`):
//   int half_edge_face_index(int half_edge_index) const;
//   int half_edge_face_offset(int half_edge_index) const;
//   int half_edge_index(int face_index, int edge_index) const;
template <class MeshBase>
class Mesh : public MeshBase {
 public:
  template <typename T>
  explicit Mesh(T&& cell_array)
      : MeshBase(std::forward<T>(cell_array)) {
    BuildAdjacencyInformation();
  }

  inline int number_of_half_edges() const {
    return num_half_edges_;
  }

  std::pair<int, int> half_edge(int half_edge_index) const {
    int face_index = this->half_edge_face_index(half_edge_index);
    auto f = this->face(face_index);

    int offset = this->half_edge_face_offset(half_edge_index);

    return std::make_pair(
      f[offset], f[(offset + 1) % this->number_of_sides(face_index)]);
  }

  inline int opposite_half_edge(int half_edge_index) const {
    return opposite_half_edge_[half_edge_index];
  }

  inline const std::vector<int>& half_edges_from_vertex(
      int vertex_index) const {
    return vertex_to_half_edges_[vertex_index];
  }

  std::vector<int> FacesAtVertex(int vertex_index) const {
    auto& half_edges = half_edges_from_vertex(vertex_index);

    std::vector<int> faces;
    faces.reserve(half_edges.size());
    for (int half_edge_index : half_edges) {
      faces.push_back(this->half_edge_face_index(half_edge_index));
    }

    return faces;
  }

  std::vector<int> AdjacentVertices(int vertex_index,
                                    bool include_source = false) const {
    std::vector<int> adj_vertices;
    if (include_source) {
      adj_vertices.push_back(vertex_index);
    }

    for (int i : half_edges_from_vertex(vertex_index)) {
      adj_vertices.push_back(half_edge(i).second);
    }

    auto j = boundary_vertices_.find(vertex_index);
    if (j != boundary_vertices_.end()) {
      std::copy(j->second.begin(),
                j->second.end(),
                std::back_inserter(adj_vertices));
    }
    return adj_vertices;
  }

  int adjacent_face_index(int face_index, int edge_index) const {
    int half_edge_index = this->half_edge_index(face_index, edge_index);
    int opposite_index = opposite_half_edge_[half_edge_index];

    return opposite_index != -1 ?
      this->half_edge_face_index(opposite_index) : -1;
  }

  std::vector<int> NRing(int vertex_index, int N,
                         bool include_source = true) const {
    if (N == 1) {
      return AdjacentVertices(vertex_index, include_source);
    }

    std::vector<int> nring;
    std::vector<unsigned char> explored(this->number_of_vertices());
    std::fill(explored.begin(), explored.end(), 0);

    std::priority_queue<std::pair<int, int>,
                        std::vector<std::pair<int, int>>,
                        NRingComparison> pq;
    pq.push(std::make_pair(0, vertex_index));

    bool past_first = false;
    while (pq.size() > 0) {
      auto& next = pq.top();
      int next_vertex = next.second;
      int depth = next.first;
      pq.pop();

      if (depth > N) {
        break;
      }

      if (explored[next_vertex]) {
        continue;
      }

      if (past_first || include_source) {
        nring.push_back(next_vertex);
      }
      past_first = true;

      for (int i : AdjacentVertices(next_vertex)) {
        if (explored[i]) {
          continue;
        }
        pq.push(std::make_pair(depth + 1, i));
      }
      explored[next_vertex] = 1;
    }

    return nring;
  }

  bool is_vertex_closed(int vertex_index) const {
    for (int half_edge_index : half_edges_from_vertex(vertex_index)) {
      if (opposite_half_edge_[half_edge_index] == -1) {
        return false;
      }
    }
    return true;
  }

  std::vector<int> Edges(const std::vector<int>& face_indices) const {
    std::vector<int> edges;
    for (int i : face_indices) {
      for (int j = 0; j < this->number_of_sides(i); ++j) {
        int half_edge_index = this->half_edge_index(i, j);
        int opposite_index = opposite_half_edge_[half_edge_index];
        if (opposite_index == -1 || half_edge_index < opposite_index) {
          edges.push_back(half_edge_index);
        } else {
          edges.push_back(opposite_index);
        }
      }
    }

    std::sort(edges.begin(), edges.end());
    edges.resize(std::distance(edges.begin(), std::unique(edges.begin(),
                                                          edges.end())));
    return edges;
  }

  class EdgeIterator {
   public:
    EdgeIterator(const Mesh& mesh, int i=0)
      : i_(i), mesh_(mesh)
    {}

    EdgeIterator(const EdgeIterator& r)
      : i_(r.i_), mesh_(r.mesh_)
    {}

    inline EdgeIterator& operator++() {
      ++i_;
      return *this;
    }

    inline EdgeIterator operator++(int) {
      EdgeIterator i(*this);
      operator++();
      return i;
    }

    inline bool operator==(const EdgeIterator & r) const {
      return (&mesh_ == &r.mesh_) && (i_ == r.i_);
    }

    inline bool operator!=(const EdgeIterator & r) const {
      return !operator==(r);
    }

    inline std::pair<int, int> operator*() const {
      return mesh_.half_edge(i_);
    }

   private:
    int i_;
    const Mesh & mesh_;
  };

  inline std::pair<EdgeIterator, EdgeIterator> iterate_half_edges() const {
    return std::make_pair(EdgeIterator(*this),
                          EdgeIterator(*this, num_half_edges_));
  }

 private:
  void BuildAdjacencyInformation() {
    this->EnsureVertices();
    vertex_to_half_edges_.resize(this->number_of_vertices());

    std::map<std::pair<int, int>, std::vector<int>> full_edge_to_half_edges;

    // Fill `vertex_to_half_edges_` and `full_edge_to_half_edges`.
    num_half_edges_ = 0;

    for (int i = 0; i < this->number_of_faces(); ++i) {
      auto f = this->face(i);
      int n = this->number_of_sides(i);

      for (int j=0; j < n; j++) {
        ++num_half_edges_;

        int vertex_index = f[j];
        int half_edge_index = this->half_edge_index(i, j);
        vertex_to_half_edges_[vertex_index].push_back(half_edge_index);

        auto full_edge = half_edge(half_edge_index);
        if (full_edge.first > full_edge.second) {
          std::swap(full_edge.first, full_edge.second);
        }
        full_edge_to_half_edges[full_edge].push_back(half_edge_index);
      }
    }

    // Fill `opposite_half_edge_` and `boundary_vertices_`.
    opposite_half_edge_.resize(num_half_edges_);
    for (auto& i : full_edge_to_half_edges) {
      auto& half_edges = i.second;
      int l1 = half_edges[0];

      if (half_edges.size() == 1) {
        opposite_half_edge_[l1] = -1;
        auto h = half_edge(l1);
        boundary_vertices_[h.second].push_back(h.first);
      } else {
        int l2 = half_edges[1];
        opposite_half_edge_[l1] = l2;
        opposite_half_edge_[l2] = l1;
      }
    }
  }

  // NRingComparison
  struct NRingComparison {
    bool operator()(const std::pair<int, int>& l,
                    const std::pair<int, int>& r) {
      if (r.first < l.first) {
        return true;
      } else if (r.first == l.first) {
        return r.second < l.second;
      }
      return false;
    }
  };

 private:
  int num_half_edges_;
  std::vector<std::vector<int>> vertex_to_half_edges_;
  std::vector<int> opposite_half_edge_;
  std::map<int, std::vector<int>> boundary_vertices_;
};

} // namespace mesh

#endif // MESH_MESH_H
