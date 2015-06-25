#include "CostFunction.hpp"
#include "PairwiseCostFunction.hpp"
#include "PairwiseCostGradientWithRespectToParams.hpp"
#include "UnaryCostFunction.hpp"
#include "UnaryCostGradientWithRespectToParams.hpp"
#include <cusp/copy.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog/logging.h>
#include <iomanip>

template<int NumDimensions, typename IndexType, typename ValueType, typename MemorySpace>
cusp::csr_matrix<IndexType, ValueType, MemorySpace> CostFunction<NumDimensions, IndexType, ValueType, MemorySpace>::AllocateJacobian() const
{
  cusp::array1d<IndexType, cusp::host_memory> column_indices1(tildeP.size() + tildeP.size());

  const int numDimensions2 = NumDimensions + NumDimensions;

  for (int i = 0; i != column_indices1.size(); ++i)
  {
    IndexType index = i % NumDimensions + NumDimensions * (i / numDimensions2);

    if (i % numDimensions2 >= NumDimensions)
    {
      index += tildeP.size();
    }

    column_indices1[i] = index;
  }

  const int numDimensions4 = numDimensions2 + numDimensions2;

  cusp::array1d<IndexType, cusp::host_memory> column_indices2(indices1.size() * numDimensions4);

  cusp::array1d<IndexType, cusp::host_memory> hindices1(indices1);
  cusp::array1d<IndexType, cusp::host_memory> hindices2(indices2);

  for (int i = 0; i != column_indices2.size(); ++i)
  {
    IndexType k;

    if (i % numDimensions4 < numDimensions2)
    {
      k = (i % numDimensions2) + numDimensions2 * hindices1[i / numDimensions4];
    }
    else
    {
      k = (i % numDimensions2) + numDimensions2 * hindices2[i / numDimensions4];
    }

    IndexType index;

    index = k % NumDimensions + NumDimensions * (k / numDimensions2);

    if (k % numDimensions2 >= NumDimensions)
    {
      index += tildeP.size();
    }

    column_indices2[i] = index;
  }

  cusp::array1d<IndexType, cusp::host_memory> row_offsets1(tildeP.size() / NumDimensions + 1);
  cusp::array1d<IndexType, cusp::host_memory> row_offsets2(indices1.size() + 1);

  row_offsets1[0] = 0;

  for (int i = 1; i != row_offsets1.size(); ++i)
  {
    row_offsets1[i] = row_offsets1[i - 1] + numDimensions2;
  }

  row_offsets2[0] = row_offsets1[row_offsets1.size() - 1];

  for (int i = 1; i != row_offsets2.size(); ++i)
  {
    row_offsets2[i] = row_offsets2[i - 1] + numDimensions4;
  }

  cusp::csr_matrix<IndexType, ValueType, MemorySpace> jacobian(
    tildeP.size() / NumDimensions + indices1.size(),
    column_indices1.size(),
    column_indices1.size() + column_indices2.size());

  auto column_indices = jacobian.column_indices.subarray(0, column_indices1.size());
  cusp::copy(column_indices1, column_indices);

  column_indices = jacobian.column_indices.subarray(column_indices1.size(), column_indices2.size());
  cusp::copy(column_indices2, column_indices);

  auto row_offsets = jacobian.row_offsets.subarray(0, row_offsets1.size());
  cusp::copy(row_offsets1, row_offsets);

  row_offsets = jacobian.row_offsets.subarray(row_offsets1.size() - 1, row_offsets2.size());
  cusp::copy(row_offsets2, row_offsets);

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << " The Jacobian has" << std::setw(9) << jacobian.num_rows << " rows and" << std::setw(9) << jacobian.num_cols << " columns";
  LOG(INFO) << "";

  return jacobian;
}

template<int NumDimensions, typename IndexType, typename ValueType, typename MemorySpace>
void CostFunction<NumDimensions, IndexType, ValueType, MemorySpace>::ComputeJacobian(const cusp::array1d<ValueType, MemorySpace>& st, cusp::array1d<ValueType, MemorySpace>& jacobian) const
{
  auto s = st.subarray(0, tildeP.size());
  auto t = st.subarray(s.size(), tildeP.size());

  auto jacobian1 = jacobian.subarray(0, s.size() + t.size());
  auto jacobian2 = jacobian.subarray(jacobian1.size(), s.size() + t.size() + s.size() + t.size());

  cusp::array1d<ValueType, MemorySpace> e1(tildeP.size() / NumDimensions);//TODO DELETE

  UnaryCostGradientWithRespectToParams<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&beta[0]),
    thrust::raw_pointer_cast(&e1[0]),
    thrust::raw_pointer_cast(&jacobian1[0]),
    tildeP.size() / NumDimensions);

  cusp::array1d<ValueType, MemorySpace> e2(indices1.size());//TODO DELETE

  PairwiseCostGradientWithRespectToParams<ValueType, IndexType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&indices1[0]),
    thrust::raw_pointer_cast(&indices2[0]),
    thrust::raw_pointer_cast(&gamma[0]),
    thrust::raw_pointer_cast(&e2[0]),
    thrust::raw_pointer_cast(&jacobian2[0]),
    indices1.size());
}

template<int NumDimensions, typename IndexType, typename ValueType, typename MemorySpace>
void CostFunction<NumDimensions, IndexType, ValueType, MemorySpace>::ComputeResidual(const cusp::array1d<ValueType, MemorySpace>& st, cusp::array1d<ValueType, MemorySpace>& residual) const
{
  auto s = st.subarray(0, tildeP.size());
  auto t = st.subarray(s.size(), tildeP.size());

  auto residual1 = residual.subarray(0, tildeP.size() / NumDimensions);
  auto residual2 = residual.subarray(residual1.size(), indices1.size());

  UnaryCostFunction<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&beta[0]),
    thrust::raw_pointer_cast(&residual1[0]),
    residual1.size());

  PairwiseCostFunction<ValueType, IndexType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&indices1[0]),
    thrust::raw_pointer_cast(&indices2[0]),
    thrust::raw_pointer_cast(&gamma[0]),
    thrust::raw_pointer_cast(&residual2[0]),
    residual2.size());

  LOG(INFO) << "Unary cost function " << cusp::blas::dot(residual1, residual1);
  LOG(INFO) << "Pairwise cost function " << cusp::blas::dot(residual2, residual2);
}

//Explicit Instantiation
template class CostFunction<3, int, float, cusp::device_memory>;
