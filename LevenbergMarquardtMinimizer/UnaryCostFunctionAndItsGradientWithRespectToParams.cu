#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ProjectionOntoLineAndItsJacobian.cuh"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"

template<int numDims>
__device__ float UnaryCostFunctionAt(const float(&tildeP)[numDims], const float(&p)[numDims])
{
  float pMinusTildeP[numDims];
  float pMinusTildePSq = 0;

  for (int i = 0; i < numDims; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
    pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
  }

  return sqrtf(pMinusTildePSq);
}

template<int numDims>
__device__ void UnaryCostFunctionAndItsGradientWithRespectToParamsAt(const float* tildeP, const float* p, const float* jacTildeP, const float* jacP, float* pUnaryCostFunction, float* pUnaryCostGradient)
{
  float pMinusTildeP[numDims];
  float pMinusTildePSq = 0;

  for (int i = 0; i < numDims; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
    pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
  }

  float invPMinusTildeP = rsqrtf(pMinusTildePSq);
  float nablaPMinusTildeP = 0;

  for (int i = 0; i < numDims; ++i)
  {
    nablaPMinusTildeP += pMinusTildeP[i] * (jacP[i] - jacTildeP[i]) * invPMinusTildeP;
  }

  *pUnaryCostFunction = sqrtf(pMinusTildePSq);
  *pUnaryCostGradient = nablaPMinusTildeP;
}

template<int numDims>
__global__ void UnaryCostFunctionAndItsGradientWithRespectToParams(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, const float *pSigma, float* pUnaryCostFunction, float* pUnaryCostGradient)
{
  float tildeP[numDims];
  float s[numDims];
  float t[numDims];
  float p[numDims];

  const int numParams = blockDim.x;

  const int numPnt = blockIdx.x;
  const int numPar = threadIdx.x;

  const int indPnt0 = numDims * numPnt;

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildeP[i] = pTildeP[indPnt];
    s[i] = pS[indPnt];
    t[i] = pT[indPnt];
  }

  float jacTildeP[numDims];
  float jacS[numDims];
  float jacT[numDims];
  float jacP[numDims];

  const int indJac0 = numDims * numParams * numPnt;

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    jacTildeP[i] = pJacTildeP[indJac];
    jacS[i] = pJacS[indJac];
    jacT[i] = pJacT[indJac];
  }

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

  const int indGrad0 = numParams * numPnt;

  float costFunction;
  float costGradient;
  UnaryCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(tildeP, p, jacTildeP, jacP, &costFunction, &costGradient);

  const float sigma = pSigma[numPnt];
  
  const float h = 1e-2;

  if (!isfinite(costGradient / sigma))
  {
    //int i = numPar % numDims;

    //if (numPar < numDims)
    //{
    //  s[i] = pS[indPnt0 + i] + h;
    //}
    //else
    //{
    //  t[i] = pT[indPnt0 + i] + h;
    //}

    //ProjectionOntoLineAt<numDims>(tildeP, s, t, p);

    //costGradient = UnaryCostFunctionAt<numDims>(tildeP, p);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //if (numPar < numDims)
    //{
    //  s[i] = pS[indPnt0 + i] - h;
    //}
    //else
    //{
    //  t[i] = pT[indPnt0 + i] - h;
    //}

    //ProjectionOntoLineAt<numDims>(tildeP, s, t, p);

    //costGradient -= UnaryCostFunctionAt<numDims>(tildeP, p);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //costGradient /= 2 * h;

    float pMinusTildeP[numDims];
    float pMinusTildePSq = 0;

    if (numPar < numDims)
    {
      s[numPar % numDims] = pS[indPnt0 + numPar % numDims] + 2 * h;
    }
    else
    {
      t[numPar % numDims] = pT[indPnt0 + numPar % numDims] + 2 * h;
    }

    ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

    for (int i = 0; i < numDims; ++i)
    {
      pMinusTildeP[i] = p[i] - tildeP[i];
      pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
    }

    float pPlusTwoHMinusTildeP = sqrtf(pMinusTildePSq);

    if (numPar < numDims)
    {
      s[numPar % numDims] = pS[indPnt0 + numPar % numDims] + h;
    }
    else
    {
      t[numPar % numDims] = pT[indPnt0 + numPar % numDims] + h;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

    for (int i = 0; i < numDims; ++i)
    {
      pMinusTildeP[i] = p[i] - tildeP[i];
      pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
    }

    float pPlusHMinusTildeP = sqrtf(pMinusTildePSq);

    if (numPar < numDims)
    {
      s[numPar % numDims] = pS[indPnt0 + numPar % numDims] - h;
    }
    else
    {
      t[numPar % numDims] = pT[indPnt0 + numPar % numDims] - h;
    }

    ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

    for (int i = 0; i < numDims; ++i)
    {
      pMinusTildeP[i] = p[i] - tildeP[i];
      pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
    }

    float pMinusHMinusTildeP = sqrtf(pMinusTildePSq);

    if (numPar < numDims)
    {
      s[numPar % numDims] = pS[indPnt0 + numPar % numDims] - 2 * h;
    }
    else
    {
      t[numPar % numDims] = pT[indPnt0 + numPar % numDims] - 2 * h;
    }

    ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

    for (int i = 0; i < numDims; ++i)
    {
      pMinusTildeP[i] = p[i] - tildeP[i];
      pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
    }

    float pMinusTwoHMinusTildeP = sqrtf(pMinusTildePSq);

    costGradient = (-pPlusTwoHMinusTildeP + 8 * pPlusHMinusTildeP - 8 * pMinusHMinusTildeP + pMinusTwoHMinusTildeP) / (12 * h);
  }

  pUnaryCostFunction[numPnt] = costFunction / sigma;
  pUnaryCostGradient[indGrad0 + numPar] = costGradient / sigma;

  assert(isfinite(costFunction));
  assert(isfinite(costGradient));
}

extern "C" void UnaryCostFunctionAndItsGradientWithRespectToParams3x6(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, const float *pSigma, float* pUnaryCostFunction, float* pUnaryCostGradient, int numPoints)
{
  UnaryCostFunctionAndItsGradientWithRespectToParams<3> << <numPoints, 6 >> >(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pSigma, pUnaryCostFunction, pUnaryCostGradient);
}