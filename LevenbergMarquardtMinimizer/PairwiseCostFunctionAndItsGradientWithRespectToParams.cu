#include <cuda.h>
#include <cuda_runtime.h>
#include "ProjectionOntoLineAndItsJacobian.cuh"
#include "PairwiseCostFunctionAndItsGradientWithRespectToParams.h"

template<int numDims>
__device__ void PairwiseCostFunctionAndItsGradientWithRespectToParamsAt(const float* piMinusPiPrime, float piMinusPiPrimeSq, const float* piMinusPj, float piMinusPjSq, const float* jacPiMinusPiPrime, const float* jacPiMinusPj, float* pPairwiseCostFunction, float* pPairwiseCostGradient)
{
  float invPiMinusPiPrime = rsqrtf(piMinusPiPrimeSq);
  float invPiMinusPj = rsqrtf(piMinusPjSq);

  float nablaPiMinusPiPrime = 0;
  float nablaPiMinusPj = 0;
  for (int i = 0; i < numDims; ++i)
  {
    nablaPiMinusPiPrime += piMinusPiPrime[i] * jacPiMinusPiPrime[i] * invPiMinusPiPrime;
    nablaPiMinusPj += piMinusPj[i] * jacPiMinusPj[i] * invPiMinusPj;
  }

  float normPiMinusPiPrime = sqrtf(piMinusPiPrimeSq);
  float normPiMinusPj = sqrtf(piMinusPjSq);

  *pPairwiseCostFunction = fmaxf(normPiMinusPiPrime * invPiMinusPj, 0);
  *pPairwiseCostGradient = fmaxf((normPiMinusPj * nablaPiMinusPiPrime - normPiMinusPiPrime * nablaPiMinusPj) / piMinusPjSq, 0);
}

template<int numDims>
__global__ void PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations(const float* pTildeP, const float* pS, const float *pT, const int *pIndPi, const int *pIndPj, const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float *pJacTildePj, const float *pJacSj, const float *pJacTj, float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj)
{
  float tildePi[numDims];
  float si[numDims];
  float ti[numDims];
  float pi[numDims];

  const int numParams = blockDim.x;

  const int numPnt = blockIdx.x;
  const int numPar = threadIdx.x;

  int indPnt0 = numDims * pIndPi[numPnt];

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildePi[i] = pTildeP[indPnt];
    si[i] = pS[indPnt];
    ti[i] = pT[indPnt];
  }

  float jacTildePi[numDims];
  float jacSi[numDims];
  float jacTi[numDims];
  float jacPi[numDims];

  const int indJac0 = numDims * numParams * numPnt;

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    jacTildePi[i] = pJacTildePi[indJac];
    jacSi[i] = pJacSi[indJac];
    jacTi[i] = pJacTi[indJac];
  }

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildePi, si, ti, jacTildePi, jacSi, jacTi, pi, jacPi);

  float tildePj[numDims];
  float sj[numDims];
  float tj[numDims];
  float pj[numDims];

  indPnt0 = numDims * pIndPj[numPnt];

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildePj[i] = pTildeP[indPnt];
    sj[i] = pS[indPnt];
    tj[i] = pT[indPnt];
  }

  float jacTildePj[numDims];
  float jacSj[numDims];
  float jacTj[numDims];
  float jacPj[numDims];

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    jacTildePj[i] = pJacTildePj[indJac];
    jacSj[i] = pJacSj[indJac];
    jacTj[i] = pJacTj[indJac];
  }

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildePj, sj, tj, jacTildePj, jacSj, jacTj, pj, jacPj);

  float piPrime[numDims];
  float jacPiPrime[numDims];

  ProjectionOntoLineAndItsJacobianAt<numDims>(pi, sj, tj, jacPi, jacSj, jacTj, piPrime, jacPiPrime);

  float piMinusPj[numDims];
  float piMinusPjSq = 0;

  float piMinusPiPrime[numDims];
  float piMinusPiPrimeSq = 0;

  float jacPiMinusPj[numDims];
  float jacPiMinusPiPrime[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    piMinusPj[i] = pi[i] - pj[i];
    piMinusPjSq += piMinusPj[i] * piMinusPj[i];

    piMinusPiPrime[i] = pi[i] - piPrime[i];
    piMinusPiPrimeSq += piMinusPiPrime[i] * piMinusPiPrime[i];

    jacPiMinusPj[i] = jacPi[i] - jacPj[i];
    jacPiMinusPiPrime[i] = jacPi[i] - jacPiPrime[i];
  }

  const int indGrad0 = numParams * numPnt;
  PairwiseCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(piMinusPiPrime, piMinusPiPrimeSq, piMinusPj, piMinusPjSq, jacPiMinusPiPrime, jacPiMinusPj, pPairwiseCostFunctioni + numPnt, pPairwiseCostGradienti + indGrad0 + numPar);
 
  float pjPrime[numDims];
  float jacPjPrime[numDims];

  ProjectionOntoLineAndItsJacobianAt<numDims>(pj, si, ti, jacPj, jacSi, jacTi, pjPrime, jacPjPrime);

  float pjMinusPjPrime[numDims];
  float pjMinusPjPrimeSq = 0;

  float jacPjMinusPjPrime[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    pjMinusPjPrime[i] = pj[i] - pjPrime[i];
    pjMinusPjPrimeSq += pjMinusPjPrime[i] * pjMinusPjPrime[i];

    jacPjMinusPjPrime[i] = jacPj[i] - jacPjPrime[i];
  }

  PairwiseCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(pjMinusPjPrime, pjMinusPjPrimeSq, piMinusPj, piMinusPjSq, jacPjMinusPjPrime, jacPiMinusPj, pPairwiseCostFunctionj + numPnt, pPairwiseCostGradientj + indGrad0 + numPar);
}


template<int numDims>
__global__ void PairwiseCostFunctionAndItsGradientWithRespectToParams(const float* pTildePi, const float* pSi, const float *pTi, const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float* pTildePj, const float* pSj, const float *pTj, const float *pJacTildePj, const float *pJacSj, const float *pJacTj, float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj)
{
  float tildePi[numDims];
  float si[numDims];
  float ti[numDims];
  float pi[numDims];

  const int numParams = blockDim.x;

  const int numPnt = blockIdx.x;
  const int numPar = threadIdx.x;

  const int indPnt0 = numDims * numPnt;

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildePi[i] = pTildePi[indPnt];
    si[i] = pSi[indPnt];
    ti[i] = pTi[indPnt];
  }

  float jacTildePi[numDims];
  float jacSi[numDims];
  float jacTi[numDims];
  float jacPi[numDims];

  const int indJac0 = numDims * numParams * numPnt;

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    jacTildePi[i] = pJacTildePi[indJac];
    jacSi[i] = pJacSi[indJac];
    jacTi[i] = pJacTi[indJac];
  }

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildePi, si, ti, jacTildePi, jacSi, jacTi, pi, jacPi);

  float tildePj[numDims];
  float sj[numDims];
  float tj[numDims];
  float pj[numDims];

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildePj[i] = pTildePj[indPnt];
    sj[i] = pSj[indPnt];
    tj[i] = pTj[indPnt];
  }

  float jacTildePj[numDims];
  float jacSj[numDims];
  float jacTj[numDims];
  float jacPj[numDims];

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    jacTildePj[i] = pJacTildePj[indJac];
    jacSj[i] = pJacSj[indJac];
    jacTj[i] = pJacTj[indJac];
  }

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildePj, sj, tj, jacTildePj, jacSj, jacTj, pj, jacPj);

  float piPrime[numDims];
  float jacPiPrime[numDims];

  ProjectionOntoLineAndItsJacobianAt<numDims>(pi, sj, tj, jacPi, jacSj, jacTj, piPrime, jacPiPrime);

  float piMinusPj[numDims];
  float piMinusPjSq = 0;

  float piMinusPiPrime[numDims];
  float piMinusPiPrimeSq = 0;

  float jacPiMinusPj[numDims];
  float jacPiMinusPiPrime[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    piMinusPj[i] = pi[i] - pj[i];
    piMinusPjSq += piMinusPj[i] * piMinusPj[i];

    piMinusPiPrime[i] = pi[i] - piPrime[i];
    piMinusPiPrimeSq += piMinusPiPrime[i] * piMinusPiPrime[i];

    jacPiMinusPj[i] = jacPi[i] - jacPj[i];
    jacPiMinusPiPrime[i] = jacPi[i] - jacPiPrime[i];
  }

  const int indGrad0 = numParams * numPnt;
  PairwiseCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(piMinusPiPrime, piMinusPiPrimeSq, piMinusPj, piMinusPjSq, jacPiMinusPiPrime, jacPiMinusPj, pPairwiseCostFunctioni + numPnt, pPairwiseCostGradienti + indGrad0 + numPar);

  float pjPrime[numDims];
  float jacPjPrime[numDims];

  ProjectionOntoLineAndItsJacobianAt<numDims>(pj, si, ti, jacPj, jacSi, jacTi, pjPrime, jacPjPrime);

  float pjMinusPjPrime[numDims];
  float pjMinusPjPrimeSq = 0;

  float jacPjMinusPjPrime[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    pjMinusPjPrime[i] = pj[i] - pjPrime[i];
    pjMinusPjPrimeSq += pjMinusPjPrime[i] * pjMinusPjPrime[i];

    jacPjMinusPjPrime[i] = jacPj[i] - jacPjPrime[i];
  }

  PairwiseCostFunctionAndItsGradientWithRespectToParamsAt<numDims>(pjMinusPjPrime, pjMinusPjPrimeSq, piMinusPj, piMinusPjSq, jacPjMinusPjPrime, jacPiMinusPj, pPairwiseCostFunctionj + numPnt, pPairwiseCostGradientj + indGrad0 + numPar);
}

extern "C" void PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(const float* pTildePi, const float* pSi, const float *pTi, const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float* pTildePj, const float* pSj, const float *pTj, const float *pJacTildePj, const float *pJacSj, const float *pJacTj, float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj, int numPoints)
{
  PairwiseCostFunctionAndItsGradientWithRespectToParams<3><<<numPoints, 12>>>(pTildePi, pSi, pTi, pJacTildePi, pJacSi, pJacTi, pTildePj, pSj, pTj, pJacTildePj, pJacSj, pJacTj, pPairwiseCostFunctioni, pPairwiseCostFunctionj, pPairwiseCostGradienti, pPairwiseCostGradientj);
}

extern "C" void PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations3x12(const float* pTildeP, const float* pS, const float *pT, const int *pIndPi, const int *pIndPj, const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float *pJacTildePj, const float *pJacSj, const float *pJacTj, float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj, int numPoints)
{
  PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations<3><<<numPoints, 12>>>(pTildeP, pS, pT, pIndPi, pIndPj, pJacTildePi, pJacSi, pJacTi, pJacTildePj, pJacSj, pJacTj, pPairwiseCostFunctioni, pPairwiseCostFunctionj, pPairwiseCostGradienti, pPairwiseCostGradientj);
}