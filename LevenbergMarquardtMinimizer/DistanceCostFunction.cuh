#ifndef DistanceCostFunction_cuh
#define DistanceCostFunction_cuh

#include "DotProduct.cuh"
#include "ProjectionOntoLine.cuh"
#include <cmath>
#include <host_defines.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, int NumDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type>
__host__ __device__ ResultType DistanceCostFunctionAt(const Arg1ValueType(&tildeP)[NumDimensions], const Arg2ValueType(&s)[NumDimensions], const Arg3ValueType(&t)[NumDimensions])
{
  ResultType p[NumDimensions];

  ProjectionOntoLineAt(tildeP, s, t, p);

  ResultType pMinusTildeP[NumDimensions];

  for (int i = 0; i != NumDimensions; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
  }

  auto const costSq = DotProduct(pMinusTildeP, pMinusTildeP);
  const ResultType epsilon = 1e-30;

  return sqrt(costSq + epsilon);
}

#endif//DistanceCostFunction_cuh