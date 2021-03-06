#ifndef LevenbergMarquardtMinimizer_hpp
#define LevenbergMarquardtMinimizer_hpp

#include <vector>

template<typename JacobianMatrixType, typename VariableVectorType, typename ResidualVectorType>
class CostFunction;

template<typename JacobianMatrixType, typename VariableVectorType, typename ResidualVectorType, typename ValueType = typename JacobianMatrixType::value_type>
void LevenbergMarquardtMinimizer(
  const CostFunction<JacobianMatrixType, VariableVectorType, ResidualVectorType>& func, 
  VariableVectorType& x, ValueType& damp, ValueType dampmin, ValueType tolx, ValueType tolf, ValueType tolg, int& itn, int itnlim, std::vector<ValueType>& costValue, double tau);

#include "LevenbergMarquardtMinimizer.inl"

#endif//LevenbergMarquardtMinimizer_hpp
