#ifndef PairwiseCostGradientWithRespectToParams_hpp
#define PairwiseCostGradientWithRespectToParams_hpp

#include <algorithm>
#include <cuda.h>
#include <glog/logging.h>
#include "DeviceFunctions.hpp"
#include "DualNumber.hpp"

template<typename ValueType, typename IndexType, int numDimensions>
__global__ void PairwiseCostGradientWithRespectToParamsCuda(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const IndexType* pIndPi, const IndexType* pIndPj, const ValueType* pBeta, ValueType* pCostFunction, ValueType* pCostGradient, int numFunctions)
{
  typedef DualNumber<ValueType> DualValueType;

  const int numFunction = threadIdx.x + blockDim.x * blockIdx.x;
  const int numParameters = numDimensions + numDimensions + numDimensions + numDimensions;

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

    ValueType costGradient[numParameters];

    DualValueType siBar[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      siBar[i].s = si[i];
    }

    DualValueType costFunctionBar;

    for (int i = 0; i < numDimensions; ++i)
    {
      siBar[i].sPrime = 1;

      costFunctionBar = PairwiseCostFunctionAt(tildePi, siBar, ti, tildePj, sj, tj);
      costGradient[i] = costFunctionBar.sPrime;

      siBar[i].sPrime = 0;
    }

    DualValueType tiBar[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      tiBar[i].s = ti[i];
    }

    for (int i = 0; i < numDimensions; ++i)
    {
      tiBar[i].sPrime = 1;

      costFunctionBar = PairwiseCostFunctionAt(tildePi, si, tiBar, tildePj, sj, tj);
      costGradient[i + numDimensions] = costFunctionBar.sPrime;

      tiBar[i].sPrime = 0;
    }

    DualValueType sjBar[numDimensions];

    for (int i = 0; i < numDimensions; ++i)
    {
      sjBar[i].s = sj[i];
    }

    for (int i = 0; i < numDimensions; ++i)
    {
      sjBar[i].sPrime = 1;

      costFunctionBar = PairwiseCostFunctionAt(tildePi, si, ti, tildePj, sjBar, tj);
      costGradient[i + numDimensions + numDimensions] = costFunctionBar.sPrime;

      sjBar[i].sPrime = 0;
    }

    DualValueType tjBar[numDimensions];
    
    for (int i = 0; i < numDimensions; ++i)
    {
      tjBar[i].s = tj[i];
    }

    for (int i = 0; i < numDimensions; ++i)
    {
      tjBar[i].sPrime = 1;

      costFunctionBar = PairwiseCostFunctionAt(tildePi, si, ti, tildePj, sj, tjBar);
      costGradient[i + numDimensions + numDimensions + numDimensions] = costFunctionBar.sPrime;

      tjBar[i].sPrime = 0;
    }

    ValueType costFunction = costFunctionBar.s;
    const ValueType beta = pBeta[numFunction];

    pCostFunction[numFunction] = beta * costFunction;

    for (int i = 0; i < numParameters; ++i)
    {
      pCostGradient[i + numFunction * numParameters] = beta * costGradient[i];
    }
  }
}

template<typename ValueType, typename IndexType, int numDimensions>
void PairwiseCostGradientWithRespectToParams(const ValueType* pTildeP, const ValueType* pS, const ValueType *pT, const IndexType *pIndPi, const IndexType *pIndPj, const ValueType* pBeta, ValueType* pCostFunction, ValueType* pCostGradient, int numFunctions)
{
  const int numThreadsPerBlock = 128;
  const int maxBlocksDim = 65535;
  const int numParameters = numDimensions + numDimensions + numDimensions + numDimensions;

  const int numFunctionsPerBlock = numThreadsPerBlock;

  for (int numFunctionsRemaining = numFunctions; numFunctionsRemaining > 0;)
  {
    int numBlocksRequired = (numFunctionsRemaining + numFunctionsPerBlock - 1) / numFunctionsPerBlock;//ceil(numFunctionsRemaining / numFunctionsPerBlock)
    int numBlocks = std::min(numBlocksRequired, maxBlocksDim);
    int numFunctionsProcessed = std::min(numBlocks * numFunctionsPerBlock, numFunctionsRemaining);

    PairwiseCostGradientWithRespectToParamsCuda<ValueType, IndexType, numDimensions><<<numBlocks, numFunctionsPerBlock>>>(pTildeP, pS, pT, pIndPi, pIndPj, pBeta, pCostFunction, pCostGradient, numFunctionsProcessed);

    pIndPi += numFunctionsProcessed;
    pIndPj += numFunctionsProcessed;
    pBeta += numFunctionsProcessed;
    pCostFunction += numFunctionsProcessed;
    pCostGradient += numParameters * numFunctionsProcessed;

    numFunctionsRemaining -= numFunctionsProcessed;
    LOG(INFO) << __FUNCTION__ << numDimensions << " " << numFunctions - numFunctionsRemaining << "/" << numFunctions << " functions processed";
  }
}

#endif//PairwiseCostGradientWithRespectToParams_hpp