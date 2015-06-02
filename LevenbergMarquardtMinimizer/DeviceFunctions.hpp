#ifndef DeviceFunctions_hpp
#define DeviceFunctions_hpp

#include <cuda_runtime.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, int numDimensions>
__forceinline__ __host__ __device__ auto DotProduct(const Arg1ValueType(&s)[numDimensions], const Arg2ValueType(&t)[numDimensions]) -> std::common_type<Arg1ValueType, Arg2ValueType>::type
{
  typename std::common_type<Arg1ValueType, Arg2ValueType>::type val{};

  for (int i = 0; i < numDimensions; ++i)
  {
    val += s[i] * t[i];
  }

  return val;
}

template<typename ValueType>
__forceinline__ __host__ __device__ ValueType Sqrt(const ValueType& arg)
{
  const ValueType eps{ 1e-24 };

  return sqrt(arg + eps);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename ResultValueType, int numDimensions>
__host__ __device__ void ProjectionOntoLineAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions], ResultValueType(&p)[numDimensions])
{
  typename std::common_type<Arg1ValueType, Arg2ValueType>::type sMinusTildeP[numDimensions];
  typename std::common_type<Arg2ValueType, Arg3ValueType>::type sMinusT[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    sMinusTildeP[i] = s[i] - tildeP[i];
    sMinusT[i] = s[i] - t[i];
  }

  auto numLambda = DotProduct(sMinusTildeP, sMinusT);
  auto denLambda = DotProduct(sMinusT, sMinusT);

  auto lambda = numLambda / denLambda;

  for (int i = 0; i < numDimensions; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
  }
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, int numDimensions>
__host__ __device__ auto UnaryCostFunctionAt(const Arg1ValueType(&tildeP)[numDimensions], const Arg2ValueType(&s)[numDimensions], const Arg3ValueType(&t)[numDimensions]) -> std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type
{
  typedef typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type ReturnValueType;

  ReturnValueType p[numDimensions];
  ProjectionOntoLineAt(tildeP, s, t, p);

  ReturnValueType pMinusTildeP[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
  }

  auto costFunctionSq = DotProduct(pMinusTildeP, pMinusTildeP);

  return Sqrt(costFunctionSq);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, typename Arg5ValueType, typename Arg6ValueType, int numDimensions>
__host__ __device__ auto PairwiseCostFunctionAt(const Arg1ValueType(&tildePi)[numDimensions], const Arg2ValueType(&si)[numDimensions], const Arg3ValueType(&ti)[numDimensions], const Arg4ValueType(&tildePj)[numDimensions], const Arg5ValueType(&sj)[numDimensions], const Arg6ValueType(&tj)[numDimensions]) -> std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType, Arg4ValueType, Arg5ValueType, Arg6ValueType>::type
{
  typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type pi[numDimensions];
  ProjectionOntoLineAt(tildePi, si, ti, pi);

  typename std::common_type<Arg4ValueType, Arg5ValueType, Arg6ValueType>::type pj[numDimensions];
  ProjectionOntoLineAt(tildePj, sj, tj, pj);

  typedef typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType, Arg4ValueType, Arg5ValueType, Arg6ValueType>::type ReturnValueType;

  ReturnValueType pjPrime[numDimensions];
  ProjectionOntoLineAt(pj, si, ti, pjPrime);

  ReturnValueType pjMinusPjPrime[numDimensions];
  ReturnValueType piMinusPj[numDimensions];

  for (int i = 0; i < numDimensions; ++i)
  {
    pjMinusPjPrime[i] = pj[i] - pjPrime[i];
    piMinusPj[i] = pi[i] - pj[i];
  }

  auto numCostFunctionSq = DotProduct(pjMinusPjPrime, pjMinusPjPrime);
  auto denCostFunctionSq = DotProduct(piMinusPj, piMinusPj);

  auto numCostFunction = Sqrt(numCostFunctionSq);
  auto denCostFunction = Sqrt(denCostFunctionSq);

  return numCostFunction / denCostFunction;
}

#endif//DeviceFunctions_hpp