#include "LevenbergMarquardtMinimizer.hpp"
#include "LinearCombination.hpp"
#include <cusp/array1d.h>
#include <cusp/blas/blas.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

template<typename CostFunctionType, typename ValueType, typename IndexType>
void DoCpuOrGpuLevenbergMarquardtMinimizer(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType>& tangentLinesPoints1,
  std::vector<ValueType>& tangentLinesPoints2,
  std::vector<ValueType> const& radiuses,
  std::vector<IndexType> const& indices1,
  std::vector<IndexType> const& indices2,
  std::vector<ValueType> const& lambdas,
  int maxNumberOfIterations,
  double voxelPhysicalSize)
{
  typedef typename CostFunctionType::MemorySpace MemorySpace;

  cusp::array1d<ValueType, cusp::host_memory> invRadiuses(radiuses.size());
  thrust::transform(
    radiuses.cbegin(),
    radiuses.cend(),
    invRadiuses.begin(),
    [](ValueType radius) { return ValueType(1) / radius; });

  CostFunctionType costFunction(
    cusp::make_array1d_view(measurements.cbegin(), measurements.cend()),
    cusp::make_array1d_view(indices1.cbegin(), indices1.cend()),
    cusp::make_array1d_view(indices2.cbegin(), indices2.cend()),
    cusp::make_array1d_view(invRadiuses.cbegin(), invRadiuses.cend()),
    cusp::make_array1d_view(lambdas.cbegin(), lambdas.cend()),
    voxelPhysicalSize);

  cusp::array1d<ValueType, MemorySpace> optParams(
    tangentLinesPoints1.size() + tangentLinesPoints2.size());

  cusp::copy(
    cusp::make_array1d_view(tangentLinesPoints1.cbegin(), tangentLinesPoints1.cend()),
    optParams.subarray(0, tangentLinesPoints1.size()));

  cusp::copy(
    cusp::make_array1d_view(tangentLinesPoints2.cbegin(), tangentLinesPoints2.cend()),
    optParams.subarray(tangentLinesPoints1.size(), tangentLinesPoints2.size()));

  // Local variables
  ValueType damp = 0;

  const ValueType dampmin = 1e-5;
  const ValueType tolx = 1e-6;
  const ValueType tolf = 1e-6;
  const ValueType tolg = 1e-5;

  int itn = 0;

  LevenbergMarquardtMinimizer(costFunction, optParams, damp, dampmin, tolx, tolf, tolg, itn, maxNumberOfIterations);

  cusp::copy(
    optParams.subarray(0, tangentLinesPoints1.size()),
    cusp::make_array1d_view(tangentLinesPoints1.begin(), tangentLinesPoints1.end()));

  cusp::copy(
    optParams.subarray(tangentLinesPoints1.size(), tangentLinesPoints2.size()),
    cusp::make_array1d_view(tangentLinesPoints2.begin(), tangentLinesPoints2.end()));
}

template<int NumDimensions, typename ValueType, typename IndexType>
void DoCpuLevenbergMarquardtMinimizer(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType>& tangentLinesPoints1,
  std::vector<ValueType>& tangentLinesPoints2,
  std::vector<ValueType> const& radiuses,
  std::vector<IndexType> const& indices1,
  std::vector<IndexType> const& indices2,
  std::vector<ValueType> const& lambdas,
  int maxNumberOfIterations,
  double voxelPhysicalSize)
{
  typedef CpuLinearCombination<NumDimensions, ValueType, IndexType> CostFunctionType;

  DoCpuOrGpuLevenbergMarquardtMinimizer<CostFunctionType>(
    measurements,
    tangentLinesPoints1,
    tangentLinesPoints2,
    radiuses,
    indices1,
    indices2,
    lambdas,
    maxNumberOfIterations,
    voxelPhysicalSize);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void DoGpuLevenbergMarquardtMinimizer(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType>& tangentLinesPoints1,
  std::vector<ValueType>& tangentLinesPoints2,
  std::vector<ValueType> const& radiuses,
  std::vector<IndexType> const& indices1,
  std::vector<IndexType> const& indices2,
  std::vector<ValueType> const& lambdas,
  int maxNumberOfIterations,
  double voxelPhysicalSize)
{
  typedef GpuLinearCombination<NumDimensions, ValueType, IndexType> CostFunctionType;

  DoCpuOrGpuLevenbergMarquardtMinimizer<CostFunctionType>(
    measurements,
    tangentLinesPoints1,
    tangentLinesPoints2,
    radiuses,
    indices1,
    indices2,
    lambdas,
    maxNumberOfIterations,
    voxelPhysicalSize);
}

//Explicit instantiation
#define InstantiateDoCpuLevenbergMarquardtMinimizer(NumDimensions,ValueType, IndexType) \
template void DoCpuLevenbergMarquardtMinimizer<NumDimensions>( \
  std::vector<ValueType> const&, \
  std::vector<ValueType>&, \
  std::vector<ValueType>&, \
  std::vector<ValueType> const&, \
  std::vector<IndexType> const&, \
  std::vector<IndexType> const&, \
  std::vector<ValueType> const&, \
  int, \
  double);

#define InstantiateDoGpuLevenbergMarquardtMinimizer(NumDimensions,ValueType, IndexType) \
template void DoGpuLevenbergMarquardtMinimizer<NumDimensions>( \
  std::vector<ValueType> const&, \
  std::vector<ValueType>&, \
  std::vector<ValueType>&, \
  std::vector<ValueType> const&, \
  std::vector<IndexType> const&, \
  std::vector<IndexType> const&, \
  std::vector<ValueType> const&, \
  int, \
  double);

InstantiateDoCpuLevenbergMarquardtMinimizer(3, double, int)
InstantiateDoGpuLevenbergMarquardtMinimizer(3, double, int)