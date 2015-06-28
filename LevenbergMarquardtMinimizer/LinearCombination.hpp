#ifndef LinearCombination_hpp
#define LinearCombination_hpp

#include "CostFunction.hpp"
#include <type_traits>

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
class LinearCombination : public CostFunction <
  cusp::csr_matrix<IndexType, ValueType, MemorySpace>, cusp::array1d<ValueType, MemorySpace>, cusp::array1d<ValueType, MemorySpace> >
{
public:
  typedef cusp::csr_matrix<IndexType, ValueType, MemorySpace> JacobianMatrixType;
  typedef cusp::array1d<ValueType, MemorySpace> VariableVectorType;
  typedef cusp::array1d<ValueType, MemorySpace> ResidualVectorType;

  void ComputeJacobian(const VariableVectorType& x, JacobianMatrixType& jacobian) const;
  void ComputeResidual(const VariableVectorType& x, ResidualVectorType& residual) const;

  int VariableVectorLength() const { return variableVectorLength; }
  int ResidualVectorLength() const { return residualVectorLength1 + residualVectorLength2; }
  int JacobianVectorLength() const { return jacobianVectorLength1 + jacobianVectorLength2; }

protected:
  template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5>
  LinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2)
    :
    tildeP(std::forward<Vector1>(tildeP)),
    indices1(std::forward<Vector2>(indices1)), indices2(std::forward<Vector3>(indices2)),
    weights1(std::forward<Vector4>(weights1)), weights2(std::forward<Vector5>(weights2)),
    residualVectorLength1(tildeP.size() / NumDimensions), residualVectorLength2(indices1.size()),
    jacobianVectorLength1(2 * tildeP.size()), jacobianVectorLength2(4 * NumDimensions * indices1.size()),
    variableVectorLength(2 * tildeP.size())
  {
  }

  virtual void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, int residualVectorLength) const = 0;
  virtual void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, int residualVectorLength) const = 0;

  virtual void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength) const = 0;
  virtual void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength) const = 0;

private:
  typedef cusp::array1d<ValueType, MemorySpace> JacobianVectorType;

  void ComputeJacobian(const VariableVectorType& x, JacobianVectorType& jacobian) const;

  void SetUpColumnIndices(cusp::array1d<IndexType, MemorySpace>& columnIndices) const;
  void SetUpRowPointers(cusp::array1d<IndexType, MemorySpace>& rowPointers) const;

  const cusp::array1d<ValueType, MemorySpace> tildeP;
  const cusp::array1d<IndexType, MemorySpace> indices1;
  const cusp::array1d<IndexType, MemorySpace> indices2;
  const cusp::array1d<ValueType, MemorySpace> weights1;
  const cusp::array1d<ValueType, MemorySpace> weights2;

  const int residualVectorLength1;
  const int residualVectorLength2;
  const int jacobianVectorLength1;
  const int jacobianVectorLength2;
  const int variableVectorLength;
};

template<int NumDimensions, typename ValueType, typename IndexType = int>
class cpuLinearCombination : public LinearCombination < NumDimensions, ValueType, IndexType, cusp::host_memory >
{
public:
  typedef LinearCombination<NumDimensions, ValueType, IndexType, cusp::host_memory> Parent;

  template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5>
  cpuLinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2)
    : Parent(tildeP, indices1, indices2, weights1, weights2)
  {
  }

  ~cpuLinearCombination()
  {
  }

protected:
  void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, int residualVectorLength) const;
  void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, int residualVectorLength) const;

  void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength) const;
  void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength) const;
};

template<int NumDimensions, typename ValueType, typename IndexType = int>
class gpuLinearCombination : public LinearCombination < NumDimensions, ValueType, IndexType, cusp::device_memory >
{
public:
  typedef LinearCombination<NumDimensions, ValueType, IndexType, cusp::device_memory> Parent;

  template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5>
  gpuLinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2)
    : Parent(tildeP, indices1, indices2, weights1, weights2)
  {
  }

  ~gpuLinearCombination()
  {
  }

protected:
  void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, int residualVectorLength) const;
  void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, int residualVectorLength) const;

  void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength) const;
  void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength) const;
};

#include "LinearCombination.inl"

#endif//LinearCombination_hpp
