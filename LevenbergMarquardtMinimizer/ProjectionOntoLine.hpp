#ifndef ProjectionOntoLine_hpp
#define ProjectionOntoLine_hpp

#include <algorithm>
#include <cuda.h>
#include <glog/logging.h>
#include "DeviceFunctions.hpp"

template<typename ValueType, int numDimensions>
__global__ void ProjectionOntoLineCuda(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, ValueType* pP, int numPoints)
{
  const int numPoint = threadIdx.x + blockDim.x * blockIdx.x;

  if (numPoint < numPoints)
  {
    ValueType tildeP[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tildeP[i] = pTildeP[i + numPoint * numDimensions];
    }

    ValueType s[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      s[i] = pS[i + numPoint * numDimensions];
    }

    ValueType t[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      t[i] = pT[i + numPoint * numDimensions];
    }

    ValueType p[numDimensions];
    ProjectionOntoLineAt(tildeP, s, t, p);

    for (int i = 0; i < numDimensions; ++i)
    {
      pP[i + numPoint * numDimensions] = p[i];
    }
  }
}

template<typename ValueType, int numDimensions>
void ProjectionOntoLine(const ValueType* pTildeP, const ValueType* pS, const ValueType *pT, ValueType* pP, int numPoints)
{
  const int numThreadsPerBlock = 192;
  const int maxBlocksDim = 65535;

  const int numPointsPerBlock = numThreadsPerBlock;

  for (int numPointsRemaining = numPoints; numPointsRemaining > 0;)
  {
    int numBlocksRequired = (numPointsRemaining + numPointsPerBlock - 1) / numPointsPerBlock;//ceil(numPointsRemaining / numPointsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxBlocksDim);
    int numPointsProcessed = std::min(numBlocks * numPointsPerBlock, numPointsRemaining);

    ProjectionOntoLineCuda<ValueType, numDimensions><<<numBlocks, numPointsPerBlock>>>(pTildeP, pS, pT, pP, numPointsProcessed);

    pTildeP += numDimensions * numPointsProcessed;
    pS += numDimensions * numPointsProcessed;
    pT += numDimensions * numPointsProcessed;
    pP += numDimensions * numPointsProcessed;

    numPointsRemaining -= numPointsProcessed;
    LOG(INFO) << __FUNCTION__ << numDimensions << " " << numPoints - numPointsRemaining << "/" << numPoints << " points processed";
  }
}

#endif//ProjectionOntoLine_hpp