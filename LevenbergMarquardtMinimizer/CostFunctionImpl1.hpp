#ifndef CostFunctionImpl1_hpp
#define CostFunctionImpl1_hpp

#include "CostFunction.hpp" 
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>

template<int NumDimensions, typename IndexType, typename ValueType, typename MemorySpace>
class CostFunctionImpl1 : public CostFunction<IndexType, ValueType, MemorySpace>
{
public:
  template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5>
  CostFunctionImpl1(Vector1&& tildeP, Vector2&& beta, Vector3&& gamma, Vector4&& indices1, Vector5&& indices2)
    :
    tildeP(std::forward<Vector1>(tildeP)),
    beta(std::forward<Vector2>(beta)),
    gamma(std::forward<Vector3>(gamma)),
    indices1(std::forward<Vector4>(indices1)),
    indices2(std::forward<Vector5>(indices2))
  {
  }

  void ComputeJacobian(const cusp::array1d<ValueType, MemorySpace>& x, cusp::csr_matrix<IndexType, ValueType, MemorySpace>& jacobian) const;
  void ComputeJacobian(const cusp::array1d<ValueType, MemorySpace>& x, cusp::array1d<ValueType, MemorySpace>& jacobian) const;
  void ComputeResidual(const cusp::array1d<ValueType, MemorySpace>& x, cusp::array1d<ValueType, MemorySpace>& residual) const;

private:
  cusp::csr_matrix<IndexType, ValueType, MemorySpace> AllocateJacobian() const;

  const cusp::array1d<ValueType, MemorySpace> tildeP;
  const cusp::array1d<ValueType, MemorySpace> beta;
  const cusp::array1d<ValueType, MemorySpace> gamma;
  const cusp::array1d<IndexType, MemorySpace> indices1;
  const cusp::array1d<IndexType, MemorySpace> indices2;
};

#endif//CostFunctionImpl1_hpp
