#include "LinearCombination.hpp"
#include "PairwiseCostFunction.hpp"
#include "PairwiseCostGradientWithRespectToParams.hpp"
#include "UnaryCostFunction.hpp"
#include "UnaryCostGradientWithRespectToParams.hpp"
#include <cusp/blas/blas.h>
#include <cusp/copy.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog/logging.h>
#include <iomanip>
#include <type_traits>

template<int NumDimensions, typename IndexType>
struct ColumnIndices1
{
  ColumnIndices1(IndexType delta)
    : 
    delta(delta)
  {
  }

  __host__ __device__ IndexType operator()(IndexType i) const
  {
    const int NumDimensions2 = 2 * NumDimensions;

    IndexType index = i % NumDimensions + NumDimensions * (i / NumDimensions2);

    if (i % NumDimensions2 >= NumDimensions)
    {
      index += delta;
    }

    return index;
  }

private:
  const IndexType delta;
};

template<int NumDimensions, typename IndexType>
struct ColumnIndices2
{
  template<typename Vector1, typename Vector2>
  ColumnIndices2(IndexType delta, Vector1&& indices1, Vector2&& indices2)
    :
    delta(delta),
    pIndices1(thrust::raw_pointer_cast(&indices1[0])),
    pIndices2(thrust::raw_pointer_cast(&indices2[0]))
  {
  }

  __host__ __device__ IndexType operator()(IndexType i) const
  {
    const int NumDimensions2 = 2 * NumDimensions;
    const int NumDimensions4 = 4 * NumDimensions;

    IndexType k = i % NumDimensions2;

    if (i % NumDimensions4 < NumDimensions2)
    {
      k += NumDimensions2 * pIndices1[i / NumDimensions4];
    }
    else
    {
      k += NumDimensions2 * pIndices2[i / NumDimensions4];
    }

    IndexType index;

    index = k % NumDimensions + NumDimensions * (k / NumDimensions2);

    if (k % NumDimensions2 >= NumDimensions)
    {
      index += delta;
    }

    return index;
  }

private:
  const IndexType delta;
  const IndexType* const pIndices1;
  const IndexType* const pIndices2;
};

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
void LinearCombination<NumDimensions, ValueType, IndexType, MemorySpace>::ComputeJacobian(const VariableVectorType& x, JacobianMatrixType& jacobian) const
{
  jacobian.resize(ResidualVectorLength(), VariableVectorLength(), JacobianVectorLength());

  SetUpColumnIndices(jacobian.column_indices);
  SetUpRowPointers(jacobian.row_offsets);

  ComputeJacobian(x, jacobian.values);
}

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
void LinearCombination<NumDimensions, ValueType, IndexType, MemorySpace>::SetUpColumnIndices(cusp::array1d<IndexType, MemorySpace>& columnIndices) const
{
  thrust::tabulate(columnIndices.begin(), columnIndices.begin() + jacobianVectorLength1, ColumnIndices1<NumDimensions, IndexType>(variableVectorLength / 2));
  thrust::tabulate(columnIndices.begin() + jacobianVectorLength1, columnIndices.end(), ColumnIndices2<NumDimensions, IndexType>(variableVectorLength / 2, indices1, indices2));
}

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
void LinearCombination<NumDimensions, ValueType, IndexType, MemorySpace>::SetUpRowPointers(cusp::array1d<IndexType, MemorySpace>& rowPointers) const
{
  thrust::sequence(rowPointers.begin(), rowPointers.begin() + residualVectorLength1 + 1, 0, 2 * NumDimensions);
  IndexType a = rowPointers[residualVectorLength1];
  thrust::sequence(rowPointers.begin() + residualVectorLength1, rowPointers.end(), a, 4 * NumDimensions);
}

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
void LinearCombination<NumDimensions, ValueType, IndexType, MemorySpace>::ComputeJacobian(const VariableVectorType& x, JacobianVectorType& jacobian) const
{
  const auto s = x.subarray(0, x.size() / 2);
  const auto t = x.subarray(s.size(), s.size());

  ComputeJacobian1(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&weights1[0]),
    thrust::raw_pointer_cast(&jacobian[0]),
    residualVectorLength1);

  ComputeJacobian2(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&weights2[0]),
    thrust::raw_pointer_cast(&indices1[0]),
    thrust::raw_pointer_cast(&indices2[0]),
    thrust::raw_pointer_cast(&jacobian[jacobianVectorLength1]),
    residualVectorLength2);
}

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
void LinearCombination<NumDimensions, ValueType, IndexType, MemorySpace>::ComputeResidual(const VariableVectorType& x, ResidualVectorType& residual) const
{
  const auto s = x.subarray(0, x.size() / 2);
  const auto t = x.subarray(s.size(), s.size());

  residual.resize(ResidualVectorLength());

  ComputeResidual1(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&weights1[0]),
    thrust::raw_pointer_cast(&residual[0]),
    residualVectorLength1);

  ComputeResidual2(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&weights2[0]),
    thrust::raw_pointer_cast(&indices1[0]),
    thrust::raw_pointer_cast(&indices2[0]),
    thrust::raw_pointer_cast(&residual[residualVectorLength1]),
    residualVectorLength2);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void cpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, int residualVectorLength) const
{
  //TBD
}

template<int NumDimensions, typename ValueType, typename IndexType>
void cpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, int residualVectorLength) const
{
  //TBD
}

template<int NumDimensions, typename ValueType, typename IndexType>
void gpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, int residualVectorLength) const
{
  UnaryCostFunction<ValueType, NumDimensions>(pTildeP, pS, pT, pWeights, pResidual, residualVectorLength);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void gpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, int residualVectorLength) const
{
  cusp::array1d<ValueType, cusp::device_memory> tmp(residualVectorLength);//TODO DELETE

  UnaryCostGradientWithRespectToParams<ValueType, NumDimensions>(pTildeP, pS, pT, pWeights,
    thrust::raw_pointer_cast(&tmp[0]), pJacobian, residualVectorLength);

  LOG(INFO) << "Distance cost " << cusp::blas::dot(tmp, tmp);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void cpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength) const
{
  //TBD
}

template<int NumDimensions, typename ValueType, typename IndexType>
void cpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength) const
{
  //TBD
}

template<int NumDimensions, typename ValueType, typename IndexType>
void gpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength) const
{
  PairwiseCostFunction<ValueType, IndexType, NumDimensions>(pTildeP, pS, pT, pIndices1, pIndices2, pWeights, pResidual, residualVectorLength);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void gpuLinearCombination<NumDimensions, ValueType, IndexType>::ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength) const
{
  cusp::array1d<ValueType, cusp::device_memory> tmp(residualVectorLength);//TODO DELETE

  PairwiseCostGradientWithRespectToParams<ValueType, IndexType, NumDimensions>(pTildeP, pS, pT, pIndices1, pIndices2, pWeights,
    thrust::raw_pointer_cast(&tmp[0]), pJacobian, residualVectorLength);

  LOG(INFO) << "Curvature cost " << cusp::blas::dot(tmp, tmp);
}

//Explicit Instantiation
template class cpuLinearCombination < 2, float > ;
template class cpuLinearCombination < 3, float > ;

template class gpuLinearCombination < 2, float > ;
template class gpuLinearCombination < 3, float > ;