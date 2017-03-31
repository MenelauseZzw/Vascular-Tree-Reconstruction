#ifndef DistanceCostFunction_cuh
#define DistanceCostFunction_cuh

#include "DotProduct.cuh"
#include "ProjectionOntoLine.cuh"
#include <cmath>
#include <host_defines.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, int NumDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type>
__host__ __device__ ResultType DistanceCostFunctionAt(const Arg1ValueType(&tildeP)[NumDimensions], const Arg2ValueType(&s)[NumDimensions], const Arg3ValueType(&t)[NumDimensions], Arg4ValueType voxelPhysicalSize)
{
  ResultType p[NumDimensions];

  ProjectionOntoLineAt(tildeP, s, t, p);

  ResultType pMinusTildeP[NumDimensions];

  for (int i = 0; i != NumDimensions; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
  }

  auto const distanceSq = DotProduct(pMinusTildeP, pMinusTildeP);
  auto const distance = sqrt(distanceSq);
  const ResultType halfVoxelSize = 0.5 * voxelPhysicalSize;

  return (distance < halfVoxelSize) ? ResultType(0) : (distance - halfVoxelSize);//max(0, ||lp-tildeP|| - halfVoxelSize)
}

#endif//DistanceCostFunction_cuh