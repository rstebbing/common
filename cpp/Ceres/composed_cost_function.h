////////////////////////////////////////////
// File: composed_cost_function.h         //
// Copyright Richard Stebbing 2015.       //
// Distributed under the MIT License.     //
// (See accompany file LICENSE or copy at //
//  http://opensource.org/licenses/MIT)   //
////////////////////////////////////////////

#ifndef CERES_PUBLIC_COST_FUNCTION_COMPOSITION_H_
#define CERES_PUBLIC_COST_FUNCTION_COMPOSITION_H_

#include <vector>

#include "ceres/cost_function.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"

namespace ceres {

class CERES_EXPORT ComposedCostFunction : public CostFunction {
 public:
  ComposedCostFunction(CostFunction* f,
                       Ownership ownership=TAKE_OWNERSHIP);

  virtual ~ComposedCostFunction();

  void AddInputCostFunction(CostFunction* g,
                            const std::vector<double*>& parameter_blocks,
                            Ownership ownership=TAKE_OWNERSHIP);

  void AddInputParameterBlock(double* values, int size);

  void Finalize();

  virtual bool Evaluate(const double* const* parameters,
                        double* residuals,
                        double** jacobians) const;

  const std::vector<double*>& parameter_blocks() const {
    return parameter_blocks_;
  }

 private:
  struct ComposedCostFunctionInput;

  int AddInternalParameterBlock(double* values, int size);
  bool InternalEvaluate(int input_index,
                        const double* const* parameters,
                        double* residuals,
                        double** jacobians) const;

 private:
  internal::scoped_ptr<CostFunction> f_;
  Ownership owns_f_;
  bool is_finalised_;

  std::vector<ComposedCostFunctionInput*> inputs_;
  std::vector<double*> parameter_blocks_;
  std::vector<std::vector<std::pair<int, int> > >
    parameter_to_jacobian_blocks_;
};

}  // namespace ceres

#endif  // CERES_PUBLIC_COST_FUNCTION_COMPOSITION_H_
