#include "ProjectionOntoLine.cuh"
#include <algorithm>

template<typename ValueType, int NumDimensions>
void CpuProjectionOntoLine(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType* pProjections, int numMeasurements)
{
  for (int i = 0; i != numMeasurements; ++i)
  {
    auto const& tildeP = reinterpret_cast<const ValueType(&)[NumDimensions]>(pMeasurements[i * NumDimensions]);
    auto const& s      = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints1[i * NumDimensions]);
    auto const& t      = reinterpret_cast<const ValueType(&)[NumDimensions]>(pTangentLinesPoints2[i * NumDimensions]);
    auto& p            = reinterpret_cast<ValueType(&)[NumDimensions]>(pProjections[i * NumDimensions]);

    ProjectionOntoLineAt(tildeP, s, t, p);
  }
}

template<typename ValueType, int NumDimensions>
__global__ void cudaProjectionOntoLine(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType* pProjections, int numMeasurements)
{
  const int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i < numMeasurements)
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

    ValueType p[NumDimensions];
    
    ProjectionOntoLineAt(tildeP, s, t, p);

    for (int k = 0; k != NumDimensions; ++k)
    {
      pProjections[k + i * NumDimensions] = p[k];
    }
  }
}

template<typename ValueType, int NumDimensions>
void GpuProjectionOntoLine(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType* pProjections, int numMeasurements)
{
  const int numThreadsPerBlock = 192;
  const int maxNumBlocks = 65535;

  const int numPointsPerBlock = numThreadsPerBlock;

  for (int numPointsRemaining = numMeasurements; numPointsRemaining > 0;)
  {
    int numBlocksRequired = (numPointsRemaining + numPointsPerBlock - 1) / numPointsPerBlock;//ceil(numPointsRemaining / numPointsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxNumBlocks);
    int numPointsProcessed = std::min(numBlocks * numPointsPerBlock, numPointsRemaining);

    cudaProjectionOntoLine<ValueType, NumDimensions><<<numBlocks, numPointsPerBlock>>>(pMeasurements, pTangentLinesPoints1, pTangentLinesPoints2, pProjections, numPointsProcessed);

    pMeasurements += NumDimensions * numPointsProcessed;
    pTangentLinesPoints1 += NumDimensions * numPointsProcessed;
    pTangentLinesPoints2 += NumDimensions * numPointsProcessed;
    pProjections += NumDimensions * numPointsProcessed;

    numPointsRemaining -= numPointsProcessed;
  }
}
