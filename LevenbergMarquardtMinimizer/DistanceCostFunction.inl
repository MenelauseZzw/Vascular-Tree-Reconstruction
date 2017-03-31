#include "DistanceCostFunction.cuh"
#include "DualNumber.cuh"
#include <algorithm>

template<typename ValueType, int NumDimensions>
void CpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength)
{
#pragma omp parallel for
  for (int i = 0; i < residualVectorLength; ++i)
  {
    auto const& tildeP = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[i * NumDimensions]);
    auto const& s = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[i * NumDimensions]);
    auto const& t = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[i * NumDimensions]);
    const ValueType weight = pWeights[i];

    pResidual[i] = weight * DistanceCostFunctionAt(tildeP, s, t, voxelPhysicalSize);
  }
}

template<typename ValueType, int NumDimensions>
void CpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength)
{
  const int numParametersPerResidual = NumDimensions + NumDimensions;

  DualNumber<ValueType> sBar[NumDimensions];
  DualNumber<ValueType> tBar[NumDimensions];

  ValueType gradient[numParametersPerResidual];

#pragma omp parallel for private(sBar,tBar,gradient)
  for (int i = 0; i < residualVectorLength; ++i)
  {
    auto const& tildeP = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[i * NumDimensions]);
    auto const& s = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[i * NumDimensions]);
    auto const& t = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[i * NumDimensions]);

    for (int k = 0; k != NumDimensions; ++k)
    {
      sBar[k].s = s[k];
      sBar[k].sPrime = 0;
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      sBar[k].sPrime = 1;

      gradient[k] = DistanceCostFunctionAt(tildeP, sBar, t, voxelPhysicalSize).sPrime;

      sBar[k].sPrime = 0;
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      tBar[k].s = t[k];
      tBar[k].sPrime = 0;
    }

    for (int k = 0; k != NumDimensions; ++k)
    {
      tBar[k].sPrime = 1;

      gradient[k + NumDimensions] = DistanceCostFunctionAt(tildeP, s, tBar, voxelPhysicalSize).sPrime;

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
__global__ void DistanceCostResidualKernel(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength)
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

    pResidual[i] = weight * DistanceCostFunctionAt(tildeP, s, t, voxelPhysicalSize);
  }
}

template<typename ValueType, int NumDimensions>
__global__ void DistanceCostJacobianKernel(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength)
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

        gradient[k] = DistanceCostFunctionAt(tildeP, sBar, t, voxelPhysicalSize).sPrime;

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

        gradient[k + NumDimensions] = DistanceCostFunctionAt(tildeP, s, tBar, voxelPhysicalSize).sPrime;

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
void GpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength)
{
  const int numThreadsPerBlock = 192;
  const int maxNumBlocks = 65535;

  const int numResidualsPerBlock = numThreadsPerBlock;

  for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
  {
    int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
    int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

    DistanceCostResidualKernel<ValueType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pResidual, voxelPhysicalSize, numResidualsProcessed);

    pMeasurements += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints1 += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints2 += NumDimensions * numResidualsProcessed;
    pWeights += numResidualsProcessed;
    pResidual += numResidualsProcessed;

    numResidualsRemaining -= numResidualsProcessed;
  }
}

template<typename ValueType, int NumDimensions>
void GpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength)
{
  const int numThreadsPerBlock = 128;
  const int maxNumBlocks = 65535;

  const int numResidualsPerBlock = numThreadsPerBlock;

  for (int numResidualsRemaining = residualVectorLength; numResidualsRemaining > 0;)
  {
    int numBlocksRequired = (numResidualsRemaining + numResidualsPerBlock - 1) / numResidualsPerBlock;//ceil(numResidualsRemaining / numResidualsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
    int numResidualsProcessed = std::min(numBlocks * numResidualsPerBlock, numResidualsRemaining);

    DistanceCostJacobianKernel<ValueType, NumDimensions><<<numBlocks, numResidualsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pWeights, pJacobian, voxelPhysicalSize, numResidualsProcessed);

    pMeasurements += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints1 += NumDimensions * numResidualsProcessed;
    pTangentLinesPoints2 += NumDimensions * numResidualsProcessed;
    pWeights += numResidualsProcessed;
    pJacobian += (NumDimensions + NumDimensions) * numResidualsProcessed;

    numResidualsRemaining -= numResidualsProcessed;
  }
}
