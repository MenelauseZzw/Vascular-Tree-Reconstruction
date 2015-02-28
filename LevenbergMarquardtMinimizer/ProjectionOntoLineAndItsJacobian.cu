#include <cuda.h>
#include <cuda_runtime.h>

//__host__ __device__ void ProjectionOntoLineAndItsJacobian(float3 tildeP, float3 s, float3 t, float3 jacTildeP, float3 jacS, float3 jacT, float3& p, float3& jacP)
//{
//  float3 sMinusT = s - t;
//  float sMinusTSq = dot(sMinusT, sMinusT);
//
//  float3 sMinusTildeP = s - tildeP;
//  float3 jacSMinusT = jacS - jacT;
//
//  float lambda = dot(sMinusTildeP, sMinusT) / sMinusTSq;
//
//  float nablaSDotSMinusT = dot(sMinusT, jacS);
//  nablaSDotSMinusT += dot(s, jacSMinusT);
//
//  float nablaTildePDotSMinusT = dot(sMinusT, jacTildeP);
//  nablaTildePDotSMinusT += dot(tildeP, jacSMinusT);
//
//  float nablaSMinusTSq = 2 * dot(sMinusT, jacSMinusT);
//  float nablaLambda = (nablaSDotSMinusT - lambda * nablaSMinusTSq - nablaTildePDotSMinusT) / sMinusTSq;
//
//  p = s - lambda * sMinusT;
//  jacP = jacS - lambda * jacSMinusT - sMinusT * nablaLambda;
//}

template<unsigned int numDims>
__device__ void ProjectionOntoLineAndItsJacobianAt(const float* tildeP, const float* s, const float* t, const float* jacTildeP, const float* jacS, const float* jacT, float* p, float* jacP)
{
  float sMinusT[numDims];
  float sMinusTSq = 0;
  float sMinusTildeP[numDims];
  float jacSMinusT[numDims];

  for (unsigned int i = 0; i < numDims; ++i)
  {
    sMinusT[i] = s[i] - t[i];
    sMinusTSq += sMinusT[i] * sMinusT[i];
    sMinusTildeP[i] = s[i] - tildeP[i];
    jacSMinusT[i] = jacS[i] - jacT[i];
  }

  float lambda = 0;
  float nablaSDotSMinusT = 0;
  float nablaTildePDotSMinusT = 0;
  float nablaSMinusTSq = 0;

  for (unsigned int i = 0; i < numDims; ++i)
  {
    lambda += (sMinusTildeP[i] * sMinusT[i]) / sMinusTSq;
    nablaSDotSMinusT += sMinusT[i] * jacS[i];
    nablaSDotSMinusT += s[i] * jacSMinusT[i];
    nablaTildePDotSMinusT += sMinusT[i] * jacTildeP[i];
    nablaTildePDotSMinusT += tildeP[i] * jacSMinusT[i];
    nablaSMinusTSq += 2 * sMinusT[i] * jacSMinusT[i];
  }

  float nablaLambda = (nablaSDotSMinusT - lambda * nablaSMinusTSq - nablaTildePDotSMinusT) / sMinusTSq;

  for (unsigned int i = 0; i < numDims; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
    jacP[i] = jacS[i] - lambda * jacSMinusT[i] - sMinusT[i] * nablaLambda;
  }
}

__device__ void UnaryCostFunctionAndItsGradientWithRespectToParamsAt(const float* tildeP, const float* s, const float* t, const float* jacTildeP, const float* jacS, const float* jacT, float& normPMinusTildeP, float& nablaPMinusTildeP)
{
  const unsigned int numDims = 3;//TODO

  float p[numDims];
  float jacP[numDims];

  ProjectionOntoLineAndItsJacobianAt<3>(tildeP, s, t, jacTildeP, jacS, jacT, p, jacP);

  float pMinusTildeP[numDims];
  float pMinusTildePSq = 0;

  for (unsigned int i = 0; i < numDims; ++i)
  {
    pMinusTildeP[i] = p[i] - tildeP[i];
    pMinusTildePSq += pMinusTildeP[i] * pMinusTildeP[i];
  }

  normPMinusTildeP = sqrt(pMinusTildePSq);
  float invPMinusTildeP = rsqrt(pMinusTildePSq);

  nablaPMinusTildeP = 0;
  for (unsigned int i = 0; i < numDims; ++i)
  {
    nablaPMinusTildeP += (pMinusTildeP[i] * jacP[i]) * invPMinusTildeP;
  }
}

template<unsigned int numDims, unsigned int numParams>
__global__ void ProjectionOntoLineAndItsJacobian(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, float *pP, float* pJacP)
{
  const unsigned int numPoint = blockIdx.x;
  const unsigned int numParam = threadIdx.x;

  __shared__ float tildeP[numDims];
  __shared__ float s[numDims];
  __shared__ float t[numDims];
  __shared__ float p[numDims];

  //Column-Major Order
  __shared__ float jacTildeP[numParams][numDims];
  __shared__ float jacS[numParams][numDims];
  __shared__ float jacT[numParams][numDims];
  __shared__ float jacP[numParams][numDims];

  const unsigned int indPnt = numDims * numPoint + numParam;
  if (numParam < numDims)
  {
    tildeP[numParam] = pTildeP[indPnt];
    s[numParam] = pS[indPnt];
    t[numParam] = pT[indPnt];
  }

  const unsigned int indJac0 = numParams * numDims * numPoint + numParam;
  for (unsigned int numDim = 0, indJac = indJac0; numDim < numDims; ++numDim, indJac += numParams)
  {
    jacTildeP[numParam][numDim] = pJacTildeP[indJac];
    jacS[numParam][numDim] = pJacS[indJac];
    jacT[numParam][numDim] = pJacT[indJac];
  }

  __syncthreads();

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP[numParam], jacS[numParam], jacT[numParam], p, jacP[numParam]);

  if (numParam < numDims)
  {
    pP[indPnt] = p[numParam];
  }

  for (unsigned int numDim = 0, indJac = indJac0; numDim < numDims; ++numDim, indJac += numParams)
  {
    pJacP[indJac] = jacP[numParam][numDim];
  }
}

extern "C" void ProjectionOntoLineAndItsJacobian3x6(const float* pTildeP, const float* pS, const float *pT, const float *pJacTildeP, const float *pJacS, const float *pJacT, float *pP, float* pJacP, unsigned int numPoints)
{
  ProjectionOntoLineAndItsJacobian<3, 6> << <numPoints, 6 >> >(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pP, pJacP);
}