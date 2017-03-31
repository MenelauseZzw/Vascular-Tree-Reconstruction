#include "DoProjectionOntoLine.hpp"
#include "ProjectionOntoLine.hpp"
#include <cusp/array1d.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>

template<int NumDimensions, typename ValueType>
void DoCpuProjectionOntoLine(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType> const& tangentLinesPoints1,
  std::vector<ValueType> const& tangentLinesPoints2,
  std::vector<ValueType>& positions)
{
  positions.resize(measurements.size());

  CpuProjectionOntoLine<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(measurements.data()),
    thrust::raw_pointer_cast(tangentLinesPoints1.data()),
    thrust::raw_pointer_cast(tangentLinesPoints2.data()),
    thrust::raw_pointer_cast(positions.data()),
    measurements.size() / NumDimensions);
}

template<int NumDimensions, typename ValueType>
void DoGpuProjectionOntoLine(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType> const& tangentLinesPoints1,
  std::vector<ValueType> const& tangentLinesPoints2,
  std::vector<ValueType>& positions)
{
  typedef cusp::array1d<ValueType, cusp::device_memory> ArrayType;

  ArrayType gpuMeasurements(measurements);
  ArrayType gpuTangentLinesPoints1(tangentLinesPoints1);
  ArrayType gpuTangentLinesPoints2(tangentLinesPoints2);
  ArrayType gpuPositions(measurements.size());

  GpuProjectionOntoLine<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(gpuMeasurements.data()),
    thrust::raw_pointer_cast(gpuTangentLinesPoints1.data()),
    thrust::raw_pointer_cast(gpuTangentLinesPoints2.data()),
    thrust::raw_pointer_cast(gpuPositions.data()),
    measurements.size() / NumDimensions);

  positions.resize(measurements.size());
  thrust::copy(gpuPositions.cbegin(), gpuPositions.cend(), positions.begin());
}

//Explicit instantiation
#define InstantiateDoCpuProjectionOntoLine(NumDimensions,ValueType) \
template void DoCpuProjectionOntoLine<NumDimensions>( \
  std::vector<ValueType> const&, \
  std::vector<ValueType> const&, \
  std::vector<ValueType> const&, \
  std::vector<ValueType>&);

#define InstantiateDoGpuProjectionOntoLine(NumDimensions,ValueType) \
template void DoGpuProjectionOntoLine<NumDimensions>( \
  std::vector<ValueType> const&, \
  std::vector<ValueType> const&, \
  std::vector<ValueType> const&, \
  std::vector<ValueType>&);

InstantiateDoCpuProjectionOntoLine(3, double)
InstantiateDoGpuProjectionOntoLine(3, double)
