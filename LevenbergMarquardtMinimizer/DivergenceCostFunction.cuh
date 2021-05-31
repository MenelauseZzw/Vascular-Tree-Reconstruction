#ifndef DivergenceCostFunction_cuh
#define DivergenceCostFunction_cuh

#include "DotProduct.cuh"
#include "ProjectionOntoLine.cuh"
#include <cmath>
#include <host_defines.h>
#include <type_traits>

template<typename ValueType>
static __inline__ __host__ __device__ ValueType cu_sqrt(const ValueType& a)
{
	return std::sqrt(a);
}

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, typename Arg5ValueType, typename Arg6ValueType, int NumDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType, Arg4ValueType, Arg5ValueType, Arg6ValueType>::type>
__host__ __device__ ResultType DivergenceCostFunctionAt(const Arg1ValueType(&tildePi)[NumDimensions], const Arg2ValueType(&si)[NumDimensions], const Arg3ValueType(&ti)[NumDimensions], const Arg4ValueType(&tildePj)[NumDimensions], const Arg5ValueType(&sj)[NumDimensions], const Arg6ValueType(&tj)[NumDimensions])
{
	typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type pi[NumDimensions];
	typename std::common_type<Arg4ValueType, Arg5ValueType, Arg6ValueType>::type pj[NumDimensions];
	typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type lp[NumDimensions];
	typename std::common_type<Arg4ValueType, Arg5ValueType, Arg6ValueType>::type lq[NumDimensions];

	const ResultType epsilon = 1e-30;

	ProjectionOntoLineAt(tildePi, si, ti, pi);
	ProjectionOntoLineAt(tildePj, sj, tj, pj);
	for (int k = 0; k != NumDimensions; k++)
	{
		lp[k] = ti[k] - si[k];
		lq[k] = tj[k] - sj[k];
	}

	auto const lpSq = DotProduct(lp, lp);
	auto const lqSq = DotProduct(lq, lq);
	auto const lpNorm = sqrt(lpSq + epsilon);
	auto const lqNorm = sqrt(lqSq + epsilon);

	ResultType pjMinusPi[NumDimensions];

	for (int i = 0; i != NumDimensions; ++i)
	{
		pjMinusPi[i] = pj[i] - pi[i];
	}

	auto const Costpq = DotProduct(pjMinusPi, pjMinusPi);
	auto const Costpqlq = DotProduct(lq, pjMinusPi) / (lqNorm + epsilon);
	auto const Costpqlp = DotProduct(lp, pjMinusPi) / (lpNorm + epsilon);

	auto const numCost = (Costpqlq - Costpqlp);

	return -numCost;
}

#endif
