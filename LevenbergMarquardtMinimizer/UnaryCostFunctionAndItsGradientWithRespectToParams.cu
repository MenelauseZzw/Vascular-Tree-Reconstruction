#include <cuda.h>
#include <cuda_runtime.h>
#include "ProjectionOntoLineAndItsJacobian.cuh"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"

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
  *pUnaryCostGradient = fmaxf(nablaPMinusTildeP, 0);
}

template<int numDims>
__global__ void UnaryCostFunctionAndItsGradientWithRespectToParams(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, float* pUnaryCostFunction, float* pUnaryCostGradient)
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
  UnaryCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(tildeP, p, jacTildeP, jacP, pUnaryCostFunction + numPnt, pUnaryCostGradient + indGrad0 + numPar);
}

extern "C" void UnaryCostFunctionAndItsGradientWithRespectToParams3x6(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, float* pUnaryCostFunction, float* pUnaryCostGradient, int numPoints)
{
  UnaryCostFunctionAndItsGradientWithRespectToParams<3><<<numPoints, 6>>>(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pUnaryCostFunction, pUnaryCostGradient);
}