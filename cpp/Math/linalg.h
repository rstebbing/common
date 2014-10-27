// linalg.h
#ifndef MATH_LINALG_H
#define MATH_LINALG_H

// Includes
#include <iterator>

#include "Eigen/Dense"
#include "Eigen/Eigen"
#include "Eigen/Sparse"

// linalg
namespace linalg {

// Matrix
template <typename TElem>
struct Matrix {
  typedef Eigen::Matrix<TElem,
                        Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::ColMajor> Type;
};

// Vector
template <typename TElem>
struct Vector {
  typedef Eigen::Matrix<TElem,
                        Eigen::Dynamic, 1,
                        Eigen::ColMajor> Type;
};

// MatrixMap
template <typename TElem>
struct MatrixMap {
  typedef Eigen::Map<typename Matrix<TElem>::Type > Type;
};

// VectorMap
template <typename TElem>
struct VectorMap {
  typedef Eigen::Map<typename Vector<TElem>::Type > Type;
};

// ConstMatrixMap
template <typename TElem>
struct ConstMatrixMap {
  typedef Eigen::Map<const typename Matrix<TElem>::Type > Type;
};

// ConstVectorMap
template <typename TElem>
struct ConstVectorMap {
  typedef Eigen::Map<const typename Vector<TElem>::Type > Type;
};

// MatrixArrayAdapter
template <typename TM>
class MatrixArrayAdapter {
 public:
  typedef decltype(((TM * )nullptr)->col(0)) value_type;
  typedef typename TM::Index Index;

  MatrixArrayAdapter(TM * m)
    : m_(m)
  {}

  inline auto size() const -> decltype(((TM *)nullptr)->cols()) {
    return m_->cols();
  }

  inline value_type operator[](Index i) const {
    return m_->col(i);
  }

  // TODO "Upgrade" `iterator` to RandomAccessIterator.
  class iterator : public std::iterator<std::input_iterator_tag, int, int> {
   public:
    iterator(int i, TM * m)
      : i_(i), m_(m)
    {}

    iterator(const iterator & r)
      : i_(r.i_), m_(r.m_)
    {}

    inline iterator & operator++() {
      ++i_;
      return *this;
    }

    inline iterator & operator++(int) {
      iterator r(*this);
      operator++();
      return r;
    }

    inline bool operator==(const iterator & r) const {
      return (m_ == r.m_) && (i_ == r.i_);
    }

    inline bool operator!=(const iterator & r) const {
      return !operator==(r);
    }

    inline auto operator*() -> decltype(((TM *)nullptr)->col(0)) {
      return m_->col(i_);
    }

   private:
    int i_;
    TM * m_;
  };

  iterator begin() const {
    return iterator(0, m_);
  }

  iterator end() const {
    return iterator(m_->cols(), m_);
  }

protected:
  TM * m_;
};

// CSRMatrixMap
template <typename TElem>
struct CSRMatrixMap {
  typedef Eigen::MappedSparseMatrix<TElem, Eigen::RowMajor> Type;
};

} // namespace linalg

#endif // MATH_LINALG_H
