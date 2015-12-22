////////////////////////////////////////////
// File: compose_cost_functions.h         //
// Copyright Richard Stebbing 2014.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////
#ifndef COMPOSE_COST_FUNCTIONS_H
#define COMPOSE_COST_FUNCTIONS_H

// Includes
#include <algorithm>
#include <numeric>
#include <set>
#include <vector>
#include <utility>

#include "ceres/cost_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"

// ceres_utility
namespace ceres_utility {
  using namespace ceres;

  // GeneralCostFunctionComposition
  class GeneralCostFunctionComposition : public CostFunction {
   public:
    GeneralCostFunctionComposition(const CostFunction* g, bool own_g=true)
      : g_(g), own_g_(own_g), is_finalised_(false) {
      set_num_residuals(g->num_residuals());

      // Setting block sizes is postponed until `Finalise`.
    }

    virtual ~GeneralCostFunctionComposition() {
      if (own_g_) {
        delete g_;
      }

      for (auto& f : f_) {
        if (std::get<1>(f)) {
          delete std::get<0>(f);
        }
      }
    }

    void AddInputCostFunction(const CostFunction* f,
                              const std::vector<double*>& p,
                              bool own_f=true) {
      // Get the indices into local `x` for each parameter block for `f`.
      auto& parameter_block_sizes = f->parameter_block_sizes();
      std::vector<int> x_indices;
      for (int i = 0; i < static_cast<int>(p.size()); ++i) {
        x_indices.push_back(AddInternalParameterBlock(
          p[i],
          parameter_block_sizes[i]));
      }

      // Save.
      f_.push_back(std::make_tuple(f, own_f, f->num_residuals(),
                                   std::move(x_indices)));
    }

    void AddInputParameterBlock(double* p, int block_size) {
      // Single input parameters are treated as identity functions.
      std::vector<int> x_indices(1);
      x_indices[0] = AddInternalParameterBlock(p, block_size);

      // Save.
      f_.push_back(std::make_tuple(nullptr, false, block_size,
                                   std::move(x_indices)));
    }

    int AddInternalParameterBlock(double* p, int block_size) {
      auto i = std::find(parameter_blocks_.begin(),
                         parameter_blocks_.end(),
                         p);
      int j = -1;
      if (i != parameter_blocks_.end()) {
        j = static_cast<int>(std::distance(parameter_blocks_.begin(), i));
      } else {
        parameter_blocks_.push_back(p);
        parameter_block_sizes_.push_back(block_size);
        j = static_cast<int>(parameter_blocks_.size()) - 1;
      }
      return j;
    }

    void Finalise() {
      // Finalise can only be called once.
      CHECK_EQ(is_finalised_, false);

      // Ensure that all inputs to `g_` have been added.
      CHECK_EQ(g_->parameter_block_sizes().size(), f_.size());

      // Ensure that the number of residuals for each input function
      // is the same as the dimension of the input parameter blocks
      // for `g_`.
      for (int i = 0; i < static_cast<int>(f_.size()); ++i) {
        auto f = std::get<0>(f_[i]);
        if (f == nullptr) {
          CHECK_EQ(g_->parameter_block_sizes()[i],
                   parameter_block_sizes_[std::get<3>(f_[i])[0]]);
        } else {
          CHECK_EQ(g_->parameter_block_sizes()[i], f->num_residuals());
        }
      }

      // Initialise `parameter_block_to_jacobian_block_indices_`.
      parameter_block_to_jacobian_block_indices_.resize(
        parameter_blocks_.size());
      for (int i = 0; i < static_cast<int>(f_.size()); ++i) {
        auto& block_indices = std::get<3>(f_[i]);
        for (int j = 0; j < static_cast<int>(block_indices.size()); ++j) {
          parameter_block_to_jacobian_block_indices_[
            block_indices[j]].push_back(std::make_pair(i, j));
        }
      }

      // Set parameter blocks sizes to make this a valid CostFunction.
      for (int n : parameter_block_sizes_) {
        mutable_parameter_block_sizes()->push_back(n);
      }

      is_finalised_ = true;
    }

    virtual bool Evaluate(const double* const* x, double* r,
                          double** J) const {
      CHECK_EQ(is_finalised_, true);

      // Allocate `g_x_data` and `g_x`, and set pointers in `g_x`.
      auto& g_parameter_block_sizes = g_->parameter_block_sizes();
      const int num_g_parameter_blocks = static_cast<int>(
        g_parameter_block_sizes.size());
      DCHECK_EQ(g_parameter_block_sizes.size(), f_.size());
      const int num_g_parameters = std::accumulate(
        g_parameter_block_sizes.begin(),
        g_parameter_block_sizes.end(),
        0);

      internal::FixedArray<double> g_x_data(num_g_parameters);
      internal::FixedArray<double*> g_x(num_g_parameter_blocks);
      int g_x_cursor = 0;
      for (int i = 0; i < num_g_parameter_blocks; ++i) {
        g_x[i] = &g_x_data[g_x_cursor];
        g_x_cursor += g_parameter_block_sizes[i];
      }

      // If Jacobians are not required then evaluate each internal function
      // and return the residual immediately.
      if (J == nullptr) {
        for (int i = 0; i < num_g_parameter_blocks; ++i) {
          if (!InternalEvaluate(i, x, g_x[i], nullptr)) {
            return false;
          }
        }
        return g_->Evaluate(g_x.get(), r, nullptr);
      }

      // Otherwise, determine the total size of internal jacobian blocks to
      // allocate for `J_i_data`.
      // NOTE `req_J_g` saves which Jacobians will required to be saved when
      // evaluating `g` for later.
      int J_i_data_size = 0;
      std::set<int> req_J_g;
      for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); ++i) {
        if (J[i] == nullptr) {
          continue;
        }

        for (auto& t : parameter_block_to_jacobian_block_indices_[i]) {
          J_i_data_size += std::get<2>(f_[t.first]) *
                           parameter_block_sizes_[i];
          req_J_g.insert(t.first);
        }
      }
      internal::FixedArray<double> J_i_data(J_i_data_size);

      // Allocate the vector of all pointers to internal Jacobian blocks ...
      int num_jacobian_blocks = 0;
      for (auto& f : f_) {
        num_jacobian_blocks += static_cast<int>(std::get<3>(f).size());
      }
      internal::FixedArray<double*> J_i_flat(num_jacobian_blocks);
      std::fill(J_i_flat.begin(), J_i_flat.end(), nullptr);

      // And initialise the pointer-to-pointers for each internal function.
      internal::FixedArray<double**> J_i(f_.size());
      int J_i_cursor = 0;
      for (int i = 0; i < static_cast<int>(f_.size()); ++i) {
        J_i[i] = &J_i_flat[J_i_cursor];
        J_i_cursor += static_cast<int>(std::get<3>(f_[i]).size());
      }

      // Now set the relevant pointers in `J_i` to positions in `J_i_data`.
      int J_i_data_cursor = 0;
      for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); ++i) {
        if (J[i] == nullptr) {
          continue;
        }

        for (auto& t : parameter_block_to_jacobian_block_indices_[i]) {
          J_i[t.first][t.second] = &J_i_data[J_i_data_cursor];
          J_i_data_cursor += std::get<2>(f_[t.first]) *
                             parameter_block_sizes_[i];
        }
      }

      // Evaluate the relevant residuals and Jacobians for the internal
      // functions.
      for (int i = 0; i < num_g_parameter_blocks; ++i) {
        if (!InternalEvaluate(i, x, g_x[i], J_i[i])) {
          return false;
        }
      }

      // Setup `J_g` for the Jacobians from `g`.
      int J_g_data_size = 0;
      for (int i : req_J_g) {
        J_g_data_size += g_->num_residuals() * g_parameter_block_sizes[i];
      }
      internal::FixedArray<double> J_g_data(J_g_data_size);

      int J_g_cursor = 0;
      internal::FixedArray<double*> J_g(num_g_parameter_blocks);
      std::fill(J_g.begin(), J_g.end(), nullptr);
      for (int i : req_J_g) {
        J_g[i] = &J_g_data[J_g_cursor];
        J_g_cursor += g_->num_residuals() * g_parameter_block_sizes[i];
      }

      // Evaluate the residual and `J_g`.
      if (!g_->Evaluate(g_x.get(), r, J_g.get())) {
        return false;
      }

      // Now chain-rule to fill out all derivatives.
      for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); ++i) {
        if (J[i] == nullptr) {
          continue;
        }

        MatrixRef J_o(J[i], g_->num_residuals(), parameter_block_sizes_[i]);
        J_o.setZero();
        for (auto& t : parameter_block_to_jacobian_block_indices_[i]) {
          J_o += MatrixRef(J_g[t.first],
                           g_->num_residuals(),
                           g_parameter_block_sizes[t.first]) *
                 MatrixRef(J_i[t.first][t.second],
                           g_parameter_block_sizes[t.first],
                           parameter_block_sizes_[i]);
        }
      }
      return true;
    }

    const std::vector<double*>& parameter_blocks() const {
      return parameter_blocks_;
    }

   private:
    bool InternalEvaluate(int i, const double* const* x_global,
                          double* r, double** J) const {
      auto f = std::get<0>(f_[i]);
      auto& block_indices = std::get<3>(f_[i]);

      // Handle input parameter block like an identity function.
      if (f == nullptr) {
        CHECK_EQ(block_indices.size(), 1);
        int i = block_indices[0];
        const double* x0 = x_global[i];
        const int block_size = parameter_block_sizes_[i];
        std::copy(x0, x0 + block_size, r);
        if (J != nullptr && J[0] != nullptr) {
          MatrixRef(J[0], block_size, block_size).setIdentity();
        }
        return true;
      }

      // Otherwise, build `x` and call `f`.
      CHECK_EQ(block_indices.size(), f->parameter_block_sizes().size());
      internal::FixedArray<const double*> x(block_indices.size());
      for (int i = 0; i < static_cast<int>(block_indices.size()); ++i) {
        x[i] = x_global[block_indices[i]];
      }

      return f->Evaluate(x.get(), r, J);
    }

   private:
    const CostFunction* g_;
    bool own_g_;
    bool is_finalised_;
    std::vector<std::tuple<const CostFunction*,
                           bool,
                           int,
                           std::vector<int>>> f_;
    std::vector<double*> parameter_blocks_;
    std::vector<int> parameter_block_sizes_;
    std::vector<std::vector<std::pair<int, int>>>
      parameter_block_to_jacobian_block_indices_;
  };

} // namespace ceres_utility

#endif // COMPOSE_COST_FUNCTIONS_H
