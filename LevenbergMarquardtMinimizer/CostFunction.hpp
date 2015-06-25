#ifndef CostFunction_hpp
#define CostFunction_hpp

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>

template<int NumDimensions, typename IndexType, typename ValueType, typename MemorySpace>
class CostFunction
{
public:
  template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5>
  CostFunction(Vector1&& tildeP, Vector2&& beta, Vector3&& gamma, Vector4&& indices1, Vector5&& indices2)
    :
    tildeP(std::forward<Vector1>(tildeP)),
    beta(std::forward<Vector2>(beta)),
    gamma(std::forward<Vector3>(gamma)),
    indices1(std::forward<Vector4>(indices1)),
    indices2(std::forward<Vector5>(indices2))
  {
  }

  cusp::csr_matrix<IndexType, ValueType, MemorySpace> AllocateJacobian() const;
  void ComputeJacobian(const cusp::array1d<ValueType, MemorySpace>& st, cusp::array1d<ValueType, MemorySpace>& jacobian) const;
  void ComputeResidual(const cusp::array1d<ValueType, MemorySpace>& st, cusp::array1d<ValueType, MemorySpace>& residual) const;

private:
  typedef cusp::array1d<ValueType, MemorySpace> ArrayType1;
  typedef cusp::array1d<IndexType, MemorySpace> ArrayType2;

  const ArrayType1 tildeP;
  const ArrayType1 beta;
  const ArrayType1 gamma;
  const ArrayType2 indices1;
  const ArrayType2 indices2;
};

#endif//CostFunction_hpp