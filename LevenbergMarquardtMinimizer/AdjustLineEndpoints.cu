#include <cuda.h>
#include <cuda_runtime.h>
#include "AdjustLineEndpoints.h"

template<int numDims>
__device__ void AdjustLineEndpointsAt(const float* tildeP, float* s, float* t)
{
  float sMinusT[numDims];
  float sMinusTSq = 0;
  float sMinusTildeP[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    sMinusT[i] = s[i] - t[i];
    sMinusTSq += sMinusT[i] * sMinusT[i];
    sMinusTildeP[i] = s[i] - tildeP[i];
  }

  float lambda = 0;

  for (int i = 0; i < numDims; ++i)
  {
    lambda += (sMinusTildeP[i] * sMinusT[i]) / sMinusTSq;
  }

  float p[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
  }

  float invSMinusT = rsqrtf(sMinusTSq);

  for (int i = 0; i < numDims; ++i)
  {
    s[i] = p[i] + sMinusT[i] * invSMinusT;
    t[i] = p[i] - sMinusT[i] * invSMinusT;
  }
}

template<int numDims>
__global__ void AdjustLineEndpoints(const float* pTildeP, float* pS, float *pT)
{
  float tildeP[numDims];
  float s[numDims];
  float t[numDims];

  const int numPnt = blockIdx.x;
  const int indPnt0 = numDims * numPnt;

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    tildeP[i] = pTildeP[indPnt];
    s[i] = pS[indPnt];
    t[i] = pT[indPnt];
  }

  AdjustLineEndpointsAt<numDims>(tildeP, s, t);

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    pS[indPnt] = s[i];
    pT[indPnt] = t[i];
  }
}


extern "C" void AdjustLineEndpoints3(const float* pTildeP, float* pS, float *pT, int numPoints)
{
  AdjustLineEndpoints<3><<<numPoints, 1>>>(pTildeP, pS, pT);
}