#include "DivergenceCostFunction.cuh"
#include "OrientedCurvatureCostFunction.cuh"
#include "CurvatureCostFunction.cuh"
#include "DualNumber.cuh"
#include <algorithm>

template<typename ValueType, typename IndexType, int NumDimensions>
void cpuCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength)
{
#pragma omp parallel for
	for (int i = 0; i < residualVectorLength; ++i)
	{
		const IndexType index1 = pPointsIndexes1[i];

		auto const& tildePi = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[index1 * NumDimensions]);
		auto const& si = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[index1 * NumDimensions]);
		auto const& ti = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[index1 * NumDimensions]);

		const IndexType index2 = pPointsIndexes2[i];

		auto const& tildePj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[index2 * NumDimensions]);
		auto const& sj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[index2 * NumDimensions]);
		auto const& tj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[index2 * NumDimensions]);

		const ValueType weight = pWeights[i];

		pResidual[i] = weight * CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tj);
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
void cpuCurvatureCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength)
{
	const int numParametersPerResidual = 4 * NumDimensions;

	DualNumber<ValueType> sBar[NumDimensions];
	DualNumber<ValueType> tBar[NumDimensions];

	ValueType gradient[numParametersPerResidual];

#pragma omp parallel for private(sBar,tBar,gradient)
	for (int i = 0; i < residualVectorLength; ++i)
	{
		IndexType const index1 = pPointsIndexes1[i];
		IndexType const index2 = pPointsIndexes2[i];

		auto const& tildePi = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[index1 * NumDimensions]);
		auto const& tildePj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[index2 * NumDimensions]);

		auto const& si = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[index1 * NumDimensions]);
		auto const& sj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[index2 * NumDimensions]);

		auto const& ti = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[index1 * NumDimensions]);
		auto const& tj = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[index2 * NumDimensions]);

		for (int k = 0; k != NumDimensions; ++k)
		{
			sBar[k].s = si[k];
			sBar[k].sPrime = 0;
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			sBar[k].sPrime = 1;

			gradient[k] = CurvatureCostFunctionAt(tildePi, sBar, ti, tildePj, sj, tj).sPrime;

			sBar[k].sPrime = 0;
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			tBar[k].s = ti[k];
			tBar[k].sPrime = 0;
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			tBar[k].sPrime = 1;

			gradient[k + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, tBar, tildePj, sj, tj).sPrime;

			tBar[k].sPrime = 0;
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			sBar[k].s = sj[k];
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			sBar[k].sPrime = 1;

			gradient[k + NumDimensions + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sBar, tj).sPrime;

			sBar[k].sPrime = 0;
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			tBar[k].s = tj[k];
		}

		for (int k = 0; k != NumDimensions; ++k)
		{
			tBar[k].sPrime = 1;

			gradient[k + NumDimensions + NumDimensions + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tBar).sPrime;

			tBar[k].sPrime = 0;
		}

		const ValueType weight = pWeights[i];

		for (int k = 0; k != numParametersPerResidual; ++k)
		{
			pJacobian[k + i * numParametersPerResidual] = weight * gradient[k];
		}
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
__global__ void cudaCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < residualVectorLength)
	{
		const IndexType index1 = pPointsIndexes1[i];

		ValueType tildePi[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePi[k] = pMeasurements[k + index1 * NumDimensions];
		}

		ValueType si[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			si[k] = pTangentLinesPoints1[k + index1 * NumDimensions];
		}

		ValueType ti[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			ti[k] = pTangentLinesPoints2[k + index1 * NumDimensions];
		}

		const IndexType index2 = pPointsIndexes2[i];

		ValueType tildePj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePj[k] = pMeasurements[k + index2 * NumDimensions];
		}

		ValueType sj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			sj[k] = pTangentLinesPoints1[k + index2 * NumDimensions];
		}

		ValueType tj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tj[k] = pTangentLinesPoints2[k + index2 * NumDimensions];
		}

		const ValueType weight = pWeights[i];

		pResidual[i] = weight * CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tj);
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
__global__ void cudaCurvatureCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength)
{
	const int numParametersPerResidual = NumDimensions + NumDimensions + NumDimensions + NumDimensions;

	const int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < residualVectorLength)
	{
		const IndexType index1 = pPointsIndexes1[i];

		ValueType tildePi[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePi[k] = pMeasurements[k + index1 * NumDimensions];
		}

		ValueType si[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			si[k] = pTangentLinesPoints1[k + index1 * NumDimensions];
		}

		ValueType ti[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			ti[k] = pTangentLinesPoints2[k + index1 * NumDimensions];
		}

		const IndexType index2 = pPointsIndexes2[i];

		ValueType tildePj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePj[k] = pMeasurements[k + index2 * NumDimensions];
		}

		ValueType sj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			sj[k] = pTangentLinesPoints1[k + index2 * NumDimensions];
		}

		ValueType tj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tj[k] = pTangentLinesPoints2[k + index2 * NumDimensions];
		}

		const ValueType weight = pWeights[i];

		ValueType gradient[numParametersPerResidual];

		{
			DualNumber<ValueType> sBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].s = si[k];
				sBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].sPrime = 1;

				gradient[k] = CurvatureCostFunctionAt(tildePi, sBar, ti, tildePj, sj, tj).sPrime;

				sBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> tBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].s = ti[k];
				tBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].sPrime = 1;

				gradient[k + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, tBar, tildePj, sj, tj).sPrime;

				tBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> sBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].s = sj[k];
				sBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].sPrime = 1;

				gradient[k + NumDimensions + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sBar, tj).sPrime;

				sBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> tBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].s = tj[k];
				tBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].sPrime = 1;

				gradient[k + NumDimensions + NumDimensions + NumDimensions] = CurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tBar).sPrime;

				tBar[k].sPrime = 0;
			}
		}

		for (int k = 0; k != numParametersPerResidual; ++k)
		{
			pJacobian[k + i * numParametersPerResidual] = weight * gradient[k];
		}
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength)
{
	const int numThreadsPerBlock = 192;
	const int maxNumBlocks = 65535;

	const int numResidualsPerBlock = numThreadsPerBlock;

	for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
	{
		int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
		int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
		int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

		cudaCurvatureCostResidual<ValueType, IndexType, NumDimensions> << <numBlocks, numResidualsPerBlock >> > (pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pPointsIndexes1, pPointsIndexes2, pResidual, residualVectorLength);

		pPointsIndexes1 += numResidualsProcessed;
		pPointsIndexes2 += numResidualsProcessed;
		pWeights += numResidualsProcessed;
		pResidual += numResidualsProcessed;

		numResidualsRemaining -= numResidualsProcessed;
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuCurvatureCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength)
{
	const int numThreadsPerBlock = 128;
	const int maxNumBlocks = 65535;

	const int numResidualsPerBlock = numThreadsPerBlock;

	for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
	{
		int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
		int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
		int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

		cudaCurvatureCostJacobian<ValueType, IndexType, NumDimensions> << <numBlocks, numResidualsPerBlock >> > (pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pPointsIndexes1, pPointsIndexes2, pJacobian, residualVectorLength);

		pPointsIndexes1 += numResidualsProcessed;
		pPointsIndexes2 += numResidualsProcessed;
		pWeights += numResidualsProcessed;
		pJacobian += (NumDimensions + NumDimensions + NumDimensions + NumDimensions) * numResidualsProcessed;

		numResidualsRemaining -= numResidualsProcessed;
	}
}

// oriented curvature and divergence terms
template<typename ValueType, typename IndexType, int NumDimensions>
__global__ void cudaPairwiseCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength, double tau)
{
	const int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < residualVectorLength)
	{
		const IndexType index1 = pPointsIndexes1[i];

		ValueType tildePi[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePi[k] = pMeasurements[k + index1 * NumDimensions];
		}

		ValueType si[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			si[k] = pTangentLinesPoints1[k + index1 * NumDimensions];
		}

		ValueType ti[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			ti[k] = pTangentLinesPoints2[k + index1 * NumDimensions];
		}

		const IndexType index2 = pPointsIndexes2[i];

		ValueType tildePj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePj[k] = pMeasurements[k + index2 * NumDimensions];
		}

		ValueType sj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			sj[k] = pTangentLinesPoints1[k + index2 * NumDimensions];
		}

		ValueType tj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tj[k] = pTangentLinesPoints2[k + index2 * NumDimensions];
		}

		ValueType weightCurv = pWeights[i] * pWeights[i];
		ValueType weightDiv = pWeights3[i] * pWeights3[i];

		ValueType epsilon = 1e-30;

		//approximation of absolute curvature
		ValueType absCurvCoeff;

		ValueType pjPrime[NumDimensions];
		ValueType pi[NumDimensions];
		ValueType pj[NumDimensions];

		ProjectionOntoLineAt(tildePi, si, ti, pi);
		ProjectionOntoLineAt(tildePj, sj, tj, pj);
		ProjectionOntoLineAt(pj, si, ti, pjPrime);

		ValueType distPiPj = 0;
		ValueType distLpPj = 0;
		for (int k = 0; k != NumDimensions; k++)
		{
			distPiPj += (pi[k] - pj[k])*(pi[k] - pj[k]);
			distLpPj += (pj[k] - pjPrime[k])*(pj[k] - pjPrime[k]);
		}

		auto const PiPj = cu_sqrt(distPiPj + epsilon);
		auto const LpPj = cu_sqrt(distLpPj + epsilon);

		absCurvCoeff = (PiPj + 1e-4) / (LpPj + 1e-4);
		//weightCurv *= absCurvCoeff;

		//condition for oriented curvature term
		ValueType thCurv = tau;
		ValueType lplqDotProduct = 0;
		ValueType lpNorm = 0;
		ValueType lqNorm = 0;
		for (int k = 0; k != NumDimensions; k++)
		{
			lplqDotProduct += (ti[k] - si[k])*(tj[k] - sj[k]);
			lpNorm += (ti[k] - si[k])*(ti[k] - si[k]);
			lqNorm += (tj[k] - sj[k])*(tj[k] - sj[k]);
		}
		lpNorm = cu_sqrt(lpNorm + epsilon);
		lqNorm = cu_sqrt(lqNorm + epsilon);
		lplqDotProduct = lplqDotProduct / (lpNorm * lqNorm);

		if (lplqDotProduct < thCurv)
		{
			pResidual[i] = weightCurv * 1.1;
		}
		else
		{
			pResidual[i] = weightCurv * OrientedCurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tj);
		}

		//condition for divergence term
		ValueType lpVec[NumDimensions];
		ValueType lqVec[NumDimensions];
		ValueType pqVec[NumDimensions];

		for (int k = 0; k != NumDimensions; k++)
		{
			lpVec[k] = ti[k] - si[k];
			lqVec[k] = tj[k] - sj[k];
			pqVec[k] = pj[k] - pi[k];
		}

		ValueType divCond = (DotProduct(lqVec, pqVec) / cu_sqrt(DotProduct(lqVec, lqVec)) - DotProduct(lpVec, pqVec) / cu_sqrt(DotProduct(lpVec, lpVec))) / DotProduct(pqVec, pqVec);
        
        ValueType baseNorm = cu_sqrt(DotProduct(pqVec, pqVec));
		if (divCond < -0.20*baseNorm)
		{
			pResidual[i] += weightDiv * DivergenceCostFunctionAt(tildePi, si, ti, tildePj, sj, tj);
			pResidual[i] = cu_sqrt(pResidual[i] + epsilon);
		}
		else
		{
			pResidual[i] = cu_sqrt(pResidual[i] + epsilon);
		}
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
__global__ void cudaPairwiseCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength, double tau)
{
	const int numParametersPerResidual = NumDimensions + NumDimensions + NumDimensions + NumDimensions; // s and t (3-dimension) for both p and q

	const int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < residualVectorLength)
	{
		const IndexType index1 = pPointsIndexes1[i];// index for point p

		ValueType tildePi[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePi[k] = pMeasurements[k + index1 * NumDimensions];
		}

		ValueType si[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			si[k] = pTangentLinesPoints1[k + index1 * NumDimensions];
		}

		ValueType ti[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			ti[k] = pTangentLinesPoints2[k + index1 * NumDimensions];
		}

		const IndexType index2 = pPointsIndexes2[i]; //index for point q

		ValueType tildePj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tildePj[k] = pMeasurements[k + index2 * NumDimensions];
		}

		ValueType sj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			sj[k] = pTangentLinesPoints1[k + index2 * NumDimensions];
		}

		ValueType tj[NumDimensions];

		for (int k = 0; k != NumDimensions; ++k)
		{
			tj[k] = pTangentLinesPoints2[k + index2 * NumDimensions];
		}

		ValueType weightCurv = pWeights[i] * pWeights[i];
		ValueType weightDiv = pWeights3[i] * pWeights3[i];

		ValueType epsilon = 1e-30;

		//approximation of absolute curvature
		ValueType absCurvCoeff;
		ValueType pjPrime[NumDimensions];
		ValueType pi[NumDimensions];
		ValueType pj[NumDimensions];

		ProjectionOntoLineAt(tildePi, si, ti, pi);
		ProjectionOntoLineAt(tildePj, sj, tj, pj);
		ProjectionOntoLineAt(pj, si, ti, pjPrime);

		ValueType distPiPj = 0;
		ValueType distLpPj = 0;
		for (int k = 0; k != NumDimensions; k++)
		{
			distPiPj += (pi[k] - pj[k])*(pi[k] - pj[k]);
			distLpPj += (pj[k] - pjPrime[k])*(pj[k] - pjPrime[k]);
		}

		distPiPj = cu_sqrt(distPiPj + epsilon);
		distLpPj = cu_sqrt(distLpPj + epsilon);

		absCurvCoeff = (distPiPj + 1e-4) / (distLpPj + 1e-4);
		//weightCurv *= absCurvCoeff;

		//condition for oriented curvature term
		ValueType thCurv = tau;
		ValueType lplqDotProduct = 0;
		ValueType lpNorm = 0;
		ValueType lqNorm = 0;
		for (int k = 0; k != NumDimensions; k++)
		{
			lplqDotProduct += (ti[k] - si[k])*(tj[k] - sj[k]);
			lpNorm += (ti[k] - si[k])*(ti[k] - si[k]);
			lqNorm += (tj[k] - sj[k])*(tj[k] - sj[k]);
		}
		lpNorm = cu_sqrt(lpNorm + epsilon);
		lqNorm = cu_sqrt(lqNorm + epsilon);
		lplqDotProduct = lplqDotProduct / (lpNorm * lqNorm);

		if (lplqDotProduct < thCurv)
		{
			weightCurv = 0.0;
		}

		//condition for divergence term
		ValueType lpVec[NumDimensions];
		ValueType lqVec[NumDimensions];
		ValueType pqVec[NumDimensions];

		for (int k = 0; k != NumDimensions; k++)
		{
			lpVec[k] = ti[k] - si[k];
			lqVec[k] = tj[k] - sj[k];
			pqVec[k] = pj[k] - pi[k];
		}

		ValueType divCond = (DotProduct(lqVec, pqVec) / cu_sqrt(DotProduct(lqVec, lqVec)) - DotProduct(lpVec, pqVec) / cu_sqrt(DotProduct(lpVec, lpVec))) / DotProduct(pqVec, pqVec);
		
		ValueType baseNorm = cu_sqrt(DotProduct(pqVec, pqVec));
		if (divCond >= -0.20*baseNorm)
		{
			weightDiv = 0.0;
		}

		//Pairwise term gradient
		ValueType gradient[numParametersPerResidual];
		{
			DualNumber<ValueType> sBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].s = si[k];
				sBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].sPrime = 1;

				auto const DualDiv = DivergenceCostFunctionAt(tildePi, sBar, ti, tildePj, sj, tj);
				auto const DualCurv = OrientedCurvatureCostFunctionAt(tildePi, sBar, ti, tildePj, sj, tj);
				auto const DualPair = sqrt(weightCurv * DualCurv + weightDiv * DualDiv + epsilon);
				gradient[k] = DualPair.sPrime;

				sBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> tBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].s = ti[k];
				tBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].sPrime = 1;

				auto const DualDiv = DivergenceCostFunctionAt(tildePi, si, tBar, tildePj, sj, tj);
				auto const DualCurv = OrientedCurvatureCostFunctionAt(tildePi, si, tBar, tildePj, sj, tj);
				auto const DualPair = sqrt(weightCurv * DualCurv + weightDiv * DualDiv + epsilon);
				gradient[k + NumDimensions] = DualPair.sPrime;

				tBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> sBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].s = sj[k];
				sBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				sBar[k].sPrime = 1;

				auto const DualDiv = DivergenceCostFunctionAt(tildePi, si, ti, tildePj, sBar, tj);
				auto const DualCurv = OrientedCurvatureCostFunctionAt(tildePi, si, ti, tildePj, sBar, tj);
				auto const DualPair = sqrt(weightCurv * DualCurv + weightDiv * DualDiv + epsilon);
				gradient[k + NumDimensions + NumDimensions] = DualPair.sPrime;

				sBar[k].sPrime = 0;
			}
		}

		{
			DualNumber<ValueType> tBar[NumDimensions];

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].s = tj[k];
				tBar[k].sPrime = 0;
			}

			for (int k = 0; k != NumDimensions; ++k)
			{
				tBar[k].sPrime = 1;

				auto const DualDiv = DivergenceCostFunctionAt(tildePi, si, ti, tildePj, sj, tBar);
				auto const DualCurv = OrientedCurvatureCostFunctionAt(tildePi, si, ti, tildePj, sj, tBar);
				auto const DualPair = sqrt(weightCurv * DualCurv + weightDiv * DualDiv + epsilon);
				gradient[k + NumDimensions + NumDimensions + NumDimensions] = DualPair.sPrime;

				tBar[k].sPrime = 0;
			}
		}

		for (int k = 0; k != numParametersPerResidual; ++k)
		{
			pJacobian[k + i * numParametersPerResidual] = gradient[k];
		}
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuPairwiseCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength, double tau)
{
	const int numThreadsPerBlock = 192;
	const int maxNumBlocks = 65535;

	const int numResidualsPerBlock = numThreadsPerBlock;

	for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
	{
		int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
		int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
		int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

		cudaPairwiseCostResidual<ValueType, IndexType, NumDimensions> << <numBlocks, numResidualsPerBlock >> > (pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pWeights3, pPointsIndexes1, pPointsIndexes2, pResidual, residualVectorLength, tau);

		pPointsIndexes1 += numResidualsProcessed;
		pPointsIndexes2 += numResidualsProcessed;
		pWeights += numResidualsProcessed;
		pWeights3 += numResidualsProcessed;
		pResidual += numResidualsProcessed;

		numResidualsRemaining -= numResidualsProcessed;
	}
}

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuPairwiseCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength, double tau)
{
	const int numThreadsPerBlock = 128;
	const int maxNumBlocks = 65535;

	const int numResidualsPerBlock = numThreadsPerBlock;

	for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
	{
		int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
		int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
		int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

		cudaPairwiseCostJacobian<ValueType, IndexType, NumDimensions> << <numBlocks, numResidualsPerBlock >> > (pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pWeights3, pPointsIndexes1, pPointsIndexes2, pJacobian, residualVectorLength, tau);

		pPointsIndexes1 += numResidualsProcessed;
		pPointsIndexes2 += numResidualsProcessed;
		pWeights += numResidualsProcessed;
		pWeights3 += numResidualsProcessed;
		pJacobian += (NumDimensions + NumDimensions + NumDimensions + NumDimensions) * numResidualsProcessed;

		numResidualsRemaining -= numResidualsProcessed;
	}
}
