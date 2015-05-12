#ifndef DeviceFunctions_hpp
#define DeviceFunctions_hpp

#include <cuda_runtime.h>

template<typename Arg1ValueType, typename Arg2ValueType, typename ReturnValue, int numDimensions>
__forceinline__ __host__ __device__ void DotProduct(const Arg1ValueType(&a)[numDimensions], const Arg2ValueType(&b)[numDimensions], ReturnValue& val)
{
  val = ReturnValue();

  for (int i = 0; i < numDimensions; ++i)
  {
    val += a[i] * b[i];
  }
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename ReturnValue, int numDimensions>
__host__ __device__ void ProjectionOntoLineAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions], ReturnValue(&p)[numDimensions])
{
  ReturnValue sMinusTildeP[numDimensions];
  ReturnValue sMinusT[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    sMinusTildeP[i] = s[i] - tildeP[i];
    sMinusT[i] = s[i] - t[i];
  }

  ReturnValue numLambda;
  DotProduct(sMinusTildeP, sMinusT, numLambda);

  ReturnValue denLambda;
  DotProduct(sMinusT, sMinusT, denLambda);

  ReturnValue lambda = numLambda / denLambda;

  for (int i = 0; i < numDimensions; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
  }
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename ReturnValue, int numDimensions>
__host__ __device__ void UnaryCostFunctionAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions], ReturnValue& costFunction)
{
  ReturnValue p[numDimensions];
  ProjectionOntoLineAt(tildeP, s, t, p);

  ReturnValue pMinusTildeP[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
  }

  ReturnValue costFunctionSq;
  DotProduct(pMinusTildeP, pMinusTildeP, costFunctionSq);

  const ReturnValue eps = 1e-24;

  costFunction = sqrt(costFunctionSq + eps);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, typename Arg5ValueType, typename Arg6ValueType, typename ReturnValue, int numDimensions>
__host__ __device__ void PairwiseCostFunctionAt(const Arg1ValueType(&tildePi)[numDimensions], const Arg2ValueType(&si)[numDimensions], const Arg3ValueType(&ti)[numDimensions], const Arg4ValueType(&tildePj)[numDimensions], const Arg5ValueType(&sj)[numDimensions], const Arg6ValueType(&tj)[numDimensions], ReturnValue& costFunction)
{
  ReturnValue pi[numDimensions];
  ProjectionOntoLineAt(tildePi, si, ti, pi);

  ReturnValue pj[numDimensions];
  ProjectionOntoLineAt(tildePj, sj, tj, pj);

  ReturnValue pjPrime[numDimensions];
  ProjectionOntoLineAt(pj, si, ti, pjPrime);

  ReturnValue pjMinusPjPrime[numDimensions];
  ReturnValue piMinusPj[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    pjMinusPjPrime[i] = pj[i] - pjPrime[i];
    piMinusPj[i] = pi[i] - pj[i];
  }

  const ReturnValue eps = 1e-24;

  ReturnValue numCostFunctionSq;
  DotProduct(pjMinusPjPrime, pjMinusPjPrime, numCostFunctionSq);

  ReturnValue denCostFunctionSq;
  DotProduct(piMinusPj, piMinusPj, denCostFunctionSq);

  ReturnValue numCostFunction = sqrt(numCostFunctionSq + eps);
  ReturnValue denCostFunction = sqrt(denCostFunctionSq + eps);

  costFunction = numCostFunction / denCostFunction;
}

#endif//DeviceFunctions_hpp