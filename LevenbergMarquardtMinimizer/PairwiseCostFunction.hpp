#ifndef PairwiseCostFunction_hpp
#define PairwiseCostFunction_hpp

#include <algorithm>
#include <cuda.h>
#include <glog/logging.h>
#include "DeviceFunctions.hpp"

template<typename ValueType, typename IndexType, int numDimensions>
__global__ void PairwiseCostFunctionCuda(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const IndexType* pIndPi, const IndexType* pIndPj, const ValueType* pBeta, ValueType* pCostFunction, int numFunctions)
{
  const int numFunction = threadIdx.x + blockDim.x * blockIdx.x;

  if (numFunction < numFunctions)
  {
    const int indPi = pIndPi[numFunction];

    ValueType tildePi[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tildePi[i] = pTildeP[i + indPi * numDimensions];
    }

    ValueType si[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      si[i] = pS[i + indPi * numDimensions];
    }

    ValueType ti[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      ti[i] = pT[i + indPi * numDimensions];
    }

    const int indPj = pIndPj[numFunction];

    ValueType tildePj[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tildePj[i] = pTildeP[i + indPj * numDimensions];
    }

    ValueType sj[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      sj[i] = pS[i + indPj * numDimensions];
    }

    ValueType tj[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tj[i] = pT[i + indPj * numDimensions];
    }

    ValueType costFunction;
    PairwiseCostFunctionAt(tildePi, si, ti, tildePj, sj, tj, costFunction);

    const ValueType beta = pBeta[numFunction];

    pCostFunction[numFunction] = beta * costFunction;
  }
}

template<typename ValueType, typename IndexType, int numDimensions>
void PairwiseCostFunction(const ValueType* pTildeP, const ValueType* pS, const ValueType *pT, const IndexType *pIndPi, const IndexType *pIndPj, const ValueType* pBeta, ValueType* pCostFunction, int numFunctions)
{
  const int numThreadsPerBlock = 192;
  const int maxBlocksDim = 65535;

  const int numFunctionsPerBlock = numThreadsPerBlock;

  for (int numFunctionsRemaining = numFunctions; numFunctionsRemaining > 0;)
  {
    int numBlocksRequired = (numFunctionsRemaining + numFunctionsPerBlock - 1) / numFunctionsPerBlock;//ceil(numFunctionsRemaining / numFunctionsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxBlocksDim);
    int numFunctionsProcessed = std::min(numBlocks * numFunctionsPerBlock, numFunctionsRemaining);

    PairwiseCostFunctionCuda<ValueType, IndexType, numDimensions><<<numBlocks, numFunctionsPerBlock>>>(pTildeP, pS, pT, pIndPi, pIndPj, pBeta, pCostFunction, numFunctionsProcessed);

    pIndPi += numFunctionsProcessed;
    pIndPj += numFunctionsProcessed;
    pBeta += numFunctionsProcessed;
    pCostFunction += numFunctionsProcessed;

    numFunctionsRemaining -= numFunctionsProcessed;
    LOG(INFO) << __FUNCTION__ << numDimensions << " " << numFunctions - numFunctionsRemaining << "/" << numFunctions << " functions processed";
  }
}

#endif//PairwiseCostFunction_hpp