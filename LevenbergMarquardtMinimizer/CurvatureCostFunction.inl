#include "CurvatureCostFunction.cuh"
#include "DualNumber.cuh"
#include <algorithm>

template<typename ValueType, typename IndexType, int NumDimensions>
void cpuCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength)
{
  for (int i = 0; i != residualVectorLength; ++i)
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

  for (int i = 0; i != residualVectorLength; ++i)
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

    cudaCurvatureCostResidual<ValueType, IndexType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pPointsIndexes1, pPointsIndexes2, pResidual, residualVectorLength);

    pPointsIndexes1 += numResidualsProcessed;
    pPointsIndexes2 += numResidualsProcessed;
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

    cudaCurvatureCostJacobian<ValueType, IndexType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pPointsIndexes1, pPointsIndexes2, pJacobian, residualVectorLength);

    pPointsIndexes1 += numResidualsProcessed;
    pPointsIndexes2 += numResidualsProcessed;
    pJacobian += (NumDimensions + NumDimensions + NumDimensions + NumDimensions) *numResidualsProcessed;

    numResidualsRemaining -= numResidualsProcessed;
  }
}