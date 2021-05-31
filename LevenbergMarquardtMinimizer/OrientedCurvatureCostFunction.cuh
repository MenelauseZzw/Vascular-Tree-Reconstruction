#ifndef OrientedCurvatureCostFunction_cuh
#define OrientedCurvatureCostFunction_cuh

#include "DotProduct.cuh"
#include "ProjectionOntoLine.cuh"
#include <cmath>
#include <host_defines.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, typename Arg3ValueType, typename Arg4ValueType, typename Arg5ValueType, typename Arg6ValueType, int NumDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType, Arg4ValueType, Arg5ValueType, Arg6ValueType>::type>
__host__ __device__ ResultType OrientedCurvatureCostFunctionAt(const Arg1ValueType(&tildePi)[NumDimensions], const Arg2ValueType(&si)[NumDimensions], const Arg3ValueType(&ti)[NumDimensions], const Arg4ValueType(&tildePj)[NumDimensions], const Arg5ValueType(&sj)[NumDimensions], const Arg6ValueType(&tj)[NumDimensions])
{
	typename std::common_type<Arg1ValueType, Arg2ValueType, Arg3ValueType>::type pi[NumDimensions];
	typename std::common_type<Arg4ValueType, Arg5ValueType, Arg6ValueType>::type pj[NumDimensions];

	ProjectionOntoLineAt(tildePi, si, ti, pi);
	ProjectionOntoLineAt(tildePj, sj, tj, pj);

	ResultType pjPrime[NumDimensions];

	ProjectionOntoLineAt(pj, si, ti, pjPrime);

	ResultType pjMinusPjPrime[NumDimensions];
	ResultType piMinusPj[NumDimensions];

	for (int i = 0; i != NumDimensions; ++i)
	{
		pjMinusPjPrime[i] = pj[i] - pjPrime[i];
		piMinusPj[i] = pi[i] - pj[i];
	}

	auto const numCostSq = DotProduct(pjMinusPjPrime, pjMinusPjPrime);
	auto const denCostSq = DotProduct(piMinusPj, piMinusPj);

	const ResultType epsilon = 1e-30;
    
    auto const numCost = sqrt(numCostSq + epsilon);
    auto const denCost = sqrt(denCostSq + epsilon);

	return (numCost + epsilon) / (denCost + epsilon);
}

#endif
