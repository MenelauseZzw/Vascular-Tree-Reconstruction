#ifndef UnaryCostFunction_hpp
#define UnaryCostFunction_hpp

#include <algorithm>
#include <cuda.h>
#include <glog/logging.h>
#include "DeviceFunctions.hpp"

template<typename ValueType, int numDimensions>
__global__ void UnaryCostFunctionCuda(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pBeta, ValueType* pCostFunction, int numFunctions)
{
  const int numFunction = threadIdx.x + blockDim.x * blockIdx.x;

  if (numFunction < numFunctions)
  {
    ValueType tildeP[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tildeP[i] = pTildeP[i + numFunction * numDimensions];
    }

    ValueType s[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      s[i] = pS[i + numFunction * numDimensions];
    }

    ValueType t[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      t[i] = pT[i + numFunction * numDimensions];
    }

    auto costFunction = UnaryCostFunctionAt(tildeP, s, t);

    const ValueType beta = pBeta[numFunction];

    pCostFunction[numFunction] = beta * costFunction;
  }
}

template<typename ValueType, int numDimensions>
void UnaryCostFunction(const ValueType* pTildeP, const ValueType* pS, const ValueType *pT, const ValueType *pBeta, ValueType* pCostFunction, int numFunctions)
{
  const int numThreadsPerBlock = 192;
  const int maxBlocksDim = 65535;

  const int numFunctionsPerBlock = numThreadsPerBlock;

  for (int numFunctionsRemaining = numFunctions; numFunctionsRemaining > 0;)
  {
    int numBlocksRequired = (numFunctionsRemaining + numFunctionsPerBlock - 1) / numFunctionsPerBlock;//ceil(numFunctionsRemaining / numFunctionsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxBlocksDim);
    int numFunctionsProcessed = std::min(numBlocks * numFunctionsPerBlock, numFunctionsRemaining);

    UnaryCostFunctionCuda<ValueType, numDimensions><<<numBlocks, numFunctionsPerBlock>>>(pTildeP, pS, pT, pBeta, pCostFunction, numFunctionsProcessed);

    pTildeP += numDimensions * numFunctionsProcessed;
    pS += numDimensions * numFunctionsProcessed;
    pT += numDimensions * numFunctionsProcessed;
    pBeta += numFunctionsProcessed;
    pCostFunction += numFunctionsProcessed;

    numFunctionsRemaining -= numFunctionsProcessed;
    LOG(INFO) << __FUNCTION__ << numDimensions << " " << numFunctions - numFunctionsRemaining << "/" << numFunctions << " functions processed";
  }
}

#endif//UnaryCostFunction_hpp