#include "DistanceCostFunction.cuh"
#include "DualNumber.cuh"
#include <algorithm>

template<typename ValueType, int NumDimensions>
void cpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, int residualVectorLength)
{
  for (int i = 0; i != residualVectorLength; ++i)
  {
    auto const& tildeP = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[i * NumDimensions]);
    auto const& s = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[i * NumDimensions]);
    auto const& t = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[i * NumDimensions]);
    const ValueType weight = pWeights[i];

    pResidual[i] = weight * DistanceCostFunctionAt(tildeP, s, t);
  }
}

template<typename ValueType, int NumDimensions>
void cpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, int residualVectorLength)
{
  const int numParametersPerResidual = NumDimensions + NumDimensions;
  
  DualNumber<ValueType> sBar[NumDimensions];
  DualNumber<ValueType> tBar[NumDimensions];

  ValueType gradient[numParametersPerResidual];

  for (int i = 0; i != residualVectorLength; ++i)
  {
    auto const& tildeP = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[i * NumDimensions]);
    auto const& s = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[i * NumDimensions]);
    auto const& t = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[i * NumDimensions]);

    for (int k = 0; k != NumDimensions; ++k)
    {
      sBar[k].s = s[k];
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      sBar[k].sPrime = 1;

      gradient[k] = DistanceCostFunctionAt(tildeP, sBar, t).sPrime;

      sBar[k].sPrime = 0;
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      tBar[k].s = t[k];
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      tBar[k].sPrime = 1;

      gradient[k + NumDimensions] = DistanceCostFunctionAt(tildeP, s, tBar).sPrime;

      tBar[k].sPrime = 0;
    }

    const ValueType weight = pWeights[i];

    for (int k = 0; k != numParametersPerResidual; ++k)
    {
      pJacobian[k + i * numParametersPerResidual] = weight * gradient[k];
    }
  }
}

template<typename ValueType, int NumDimensions>
__global__ void cudaDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, int residualVectorLength)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < residualVectorLength)
  {
    ValueType tildeP[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      tildeP[k] = pMeasurements[k + i * NumDimensions];
    }

    ValueType s[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      s[k] = pTangentLinesPoints1[k + i * NumDimensions];
    }

    ValueType t[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      t[k] = pTangentLinesPoints2[k + i * NumDimensions];
    }

    const ValueType weight = pWeights[i];

    pResidual[i] = weight * DistanceCostFunctionAt(tildeP, s, t);
  }
}

template<typename ValueType, int NumDimensions>
__global__ void cudaDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, int residualVectorLength)
{
  const int numParametersPerResidual = NumDimensions + NumDimensions;

  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < residualVectorLength)
  {
    ValueType tildeP[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      tildeP[k] = pMeasurements[k + i * NumDimensions];
    }

    ValueType s[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      s[k] = pTangentLinesPoints1[k + i * NumDimensions];
    }

    ValueType t[NumDimensions];

    for (int k = 0; k != NumDimensions; ++k)
    {
      t[k] = pTangentLinesPoints2[k + i * NumDimensions];
    }

    ValueType gradient[numParametersPerResidual];

    {
      DualNumber<ValueType> sBar[NumDimensions];

      for (int k = 0; k != NumDimensions; ++k)
      {
        sBar[k].s = s[k];
      }

      for (int k = 0; k != NumDimensions; ++k)
      {
        sBar[k].sPrime = 1;

        gradient[k] = DistanceCostFunctionAt(tildeP, sBar, t).sPrime;

        sBar[k].sPrime = 0;
      }
    }

    {
      DualNumber<ValueType> tBar[NumDimensions];

      for (int k = 0; k != NumDimensions; ++k)
      {
        tBar[k].s = t[k];
      }

      for (int k = 0; k != NumDimensions; ++k)
      {
        tBar[k].sPrime = 1;

        gradient[k + NumDimensions] = DistanceCostFunctionAt(tildeP, s, tBar).sPrime;

        tBar[k].sPrime = 0;
      }
    }

    const ValueType weight = pWeights[i];

    for (int k = 0; k != numParametersPerResidual; ++k)
    {
      pJacobian[k + i * numParametersPerResidual] = weight * gradient[k];
    }
  }
}

template<typename ValueType, int NumDimensions>
void gpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, int residualVectorLength)
{
  const int numThreadsPerBlock = 192;
  const int maxNumBlocks = 65535;

  const int numResidualsPerBlock = numThreadsPerBlock;

  for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
  {
    int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
    int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

    cudaDistanceCostResidual<ValueType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pResidual, numResidualsProcessed);

    pMeasurements += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints1 += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints2 += NumDimensions * numResidualsProcessed;
    pWeights += numResidualsProcessed;
    pResidual += numResidualsProcessed;

    numResidualsRemaining -= numResidualsProcessed;
  }
}

template<typename ValueType, int NumDimensions>
void gpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, int residualVectorLength)
{
  const int numThreadsPerBlock = 128;
  const int maxNumBlocks = 65535;

  const int numResidualsPerBlock = numThreadsPerBlock;

  for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
  {
    int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
    int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

    cudaDistanceCostJacobian<ValueType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pJacobian, numResidualsProcessed);

    pMeasurements += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints1 += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints2 += NumDimensions * numResidualsProcessed;
    pWeights += numResidualsProcessed;
    pJacobian += (NumDimensions + NumDimensions) * numResidualsProcessed;

    numResidualsRemaining -= numResidualsProcessed;
  }
}