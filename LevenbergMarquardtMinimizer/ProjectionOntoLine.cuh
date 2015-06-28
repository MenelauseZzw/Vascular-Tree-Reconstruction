#ifndef ProjectionOntoLine_cuh
#define ProjectionOntoLine_cuh

#include "DotProduct.cuh"
#include <host_defines.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename ResultType, int NumDimensions>
__host__ __device__ void ProjectionOntoLineAt(const Arg1ValueType(&tildeP)[NumDimensions], const Arg2ValueType(&s)[NumDimensions], const Arg3ValueType(&t)[NumDimensions], ResultType(&p)[NumDimensions])
{
  typename std::common_type<Arg1ValueType, Arg2ValueType>::type sMinusTildeP[NumDimensions];
  typename std::common_type<Arg2ValueType, Arg3ValueType>::type sMinusT[NumDimensions];

  for (int i = 0; i != NumDimensions; ++i)
  {
    sMinusTildeP[i] = s[i] - tildeP[i];
    sMinusT[i] = s[i] - t[i];
  }

  auto const numLambda = DotProduct(sMinusTildeP, sMinusT);
  auto const denLambda = DotProduct(sMinusT, sMinusT);

  auto const lambda = numLambda / denLambda;

  for (int i = 0; i != NumDimensions; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
  }
}

#endif//ProjectionOntoLine_cuh
