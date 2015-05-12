#ifndef UnaryCostGradientWithRespectToParams_hpp
#define UnaryCostGradientWithRespectToParams_hpp

#include <algorithm>
#include <cuda.h>
#include <glog/logging.h>
#include "DeviceFunctions.hpp"
#include "DualNumber.hpp"

template<typename ValueType, int numDimensions>
__global__ void UnaryCostGradientWithRespectToParamsCuda(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pBeta, ValueType* pCostFunction, ValueType* pCostGradient, int numFunctions)
{
  typedef DualNumber<ValueType> DualValueType;

  const int numFunction = threadIdx.x + blockDim.x * blockIdx.x;
  const int numParameters = numDimensions + numDimensions;

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

    ValueType costGradient[numParameters];

    DualValueType sBar[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      sBar[i].s = s[i];
    }

    DualValueType costFunctionBar;

    for (int i = 0; i < numDimensions; ++i)
    {
      sBar[i].sPrime = 1;

      UnaryCostFunctionAt(tildeP, sBar, t, costFunctionBar);
      costGradient[i] = costFunctionBar.sPrime;

      sBar[i].sPrime = 0;
    }

    DualValueType tBar[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tBar[i].s = t[i];
    }

    for (int i = 0; i < numDimensions; ++i)
    {
      tBar[i].sPrime = 1;

      UnaryCostFunctionAt(tildeP, s, tBar, costFunctionBar);
      costGradient[i + numDimensions] = costFunctionBar.sPrime;

      tBar[i].sPrime = 0;
    }

    const ValueType costFunction = costFunctionBar.s;
    const ValueType beta = pBeta[numFunction];

    pCostFunction[numFunction] = beta * costFunction;

    for (int i = 0; i < numParameters; ++i)
    {
      pCostGradient[i + numFunction * numParameters] = beta * costGradient[i];
    }
  }
}

template<typename ValueType, int numDimensions>
void UnaryCostGradientWithRespectToParams(const ValueType* pTildeP, const ValueType* pS, const ValueType *pT, const ValueType *pBeta, ValueType* pCostFunction, ValueType* pCostGradient, int numFunctions)
{
  const int numThreadsPerBlock = 128;
  const int maxBlocksDim = 65535;
  const int numParameters = numDimensions + numDimensions;

  int numFunctionsPerBlock = numThreadsPerBlock;
  dim3 blockDim(numFunctionsPerBlock);

  for (int numFunctionsRemaining = numFunctions; numFunctionsRemaining > 0;)
  {
    int numBlocksRequired = (numFunctionsRemaining + numFunctionsPerBlock - 1) / numFunctionsPerBlock;//ceil(numFunctionsRemaining / numFunctionsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxBlocksDim);
    int numFunctionsProcessed = std::min(numBlocks * numFunctionsPerBlock, numFunctionsRemaining);

    UnaryCostGradientWithRespectToParamsCuda<ValueType, numDimensions><<<numBlocks, blockDim>>>(pTildeP, pS, pT, pBeta, pCostFunction, pCostGradient, numFunctionsProcessed);

    pTildeP += numDimensions * numFunctionsProcessed;
    pS += numDimensions * numFunctionsProcessed;
    pT += numDimensions * numFunctionsProcessed;
    pBeta += numFunctionsProcessed;
    pCostFunction += numFunctionsProcessed;
    pCostGradient += numParameters * numFunctionsProcessed;

    numFunctionsRemaining -= numFunctionsProcessed;
    LOG(INFO) << __FUNCTION__ << numDimensions << " " << numFunctions - numFunctionsRemaining << "/" << numFunctions << " functions processed";
  }
}

#endif//UnaryCostGradientWithRespectToParams_hpp