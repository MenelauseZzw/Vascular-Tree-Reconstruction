#ifndef DeviceFunctions_hpp
#define DeviceFunctions_hpp

#include <cuda_runtime.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, int numDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType>::type>
__forceinline__ __host__ __device__ ResultType DotProduct(const Arg1ValueType(&s)[numDimensions], const Arg2ValueType(&t)[numDimensions])
{
  ResultType val{};

  for (int i = 0; i != numDimensions; ++i)
  {
    val += s[i] * t[i];
  }

  return val;
}

template<typename ValueType>
__forceinline__ __host__ __device__ ValueType Sqrt(const ValueType& arg)
{
  const ValueType epsilon{ 1e-24 };

  return sqrt(arg + epsilon);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename ResultType, int numDimensions>
__host__ __device__ void ProjectionOntoLineAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions], ResultType(&p)[numDimensions])
{
  typename std::common_type<Arg1ValueType, Arg2ValueType>::type sMinusTildeP[numDimensions];
  typename std::common_type<Arg2ValueType, Arg3ValueType>::type sMinusT[numDimensions];

  for (int i = 0; i != numDimensions; ++i)
  {
    sMinusTildeP[i] = s[i] - tildeP[i];
    sMinusT[i] = s[i] - t[i];
  }

  auto const numLambda = DotProduct(sMinusTildeP, sMinusT);
  auto const denLambda = DotProduct(sMinusT, sMinusT);

  auto const lambda = numLambda / denLambda;

  for (int i = 0; i != numDimensions; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
  }
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, int numDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type>
__host__ __device__ ResultType UnaryCostFunctionAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions])
{
  ResultType p[numDimensions];
  ProjectionOntoLineAt(tildeP, s, t, p);

  ResultType pMinusTildeP[numDimensions];
  
  for (int i = 0; i != numDimensions; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
  }

  auto const costFunctionSq = DotProduct(pMinusTildeP, pMinusTildeP);

  return Sqrt(costFunctionSq);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, typename Arg5ValueType, typename Arg6ValueType, int numDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType, Arg4ValueType, Arg5ValueType, Arg6ValueType>::type>
__host__ __device__ ResultType PairwiseCostFunctionAt(const Arg1ValueType(&tildePi)[numDimensions], const Arg2ValueType(&si)[numDimensions], const Arg3ValueType(&ti)[numDimensions], const Arg4ValueType(&tildePj)[numDimensions], const Arg5ValueType(&sj)[numDimensions], const Arg6ValueType(&tj)[numDimensions])
{
  typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type pi[numDimensions];
  ProjectionOntoLineAt(tildePi, si, ti, pi);

  typename std::common_type<Arg4ValueType, Arg5ValueType, Arg6ValueType>::type pj[numDimensions];
  ProjectionOntoLineAt(tildePj, sj, tj, pj);
  
  ResultType pjPrime[numDimensions];
  ProjectionOntoLineAt(pj, si, ti, pjPrime);

  ResultType pjMinusPjPrime[numDimensions];
  ResultType piMinusPj[numDimensions];

  for (int i = 0; i != numDimensions; ++i)
  {
    pjMinusPjPrime[i] = pj[i] - pjPrime[i];
    piMinusPj[i] = pi[i] - pj[i];
  }

  auto const numCostFunctionSq = DotProduct(pjMinusPjPrime, pjMinusPjPrime);
  auto const denCostFunctionSq = DotProduct(piMinusPj, piMinusPj);

  auto const numCostFunction = Sqrt(numCostFunctionSq);
  auto const denCostFunction = Sqrt(denCostFunctionSq);

  return numCostFunction / denCostFunction;
}

#endif//DeviceFunctions_hpp