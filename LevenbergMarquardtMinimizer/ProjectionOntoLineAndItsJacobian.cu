#include <cuda.h>
#include <cuda_runtime.h>
#include "ProjectionOntoLineAndItsJacobian.cuh"

__constant__ static float jacTildeP[][3] =
{
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0
};

__constant__ static float jacS[][3] =
{
  1, 0, 0,
  0, 1, 0,
  0, 0, 1,
  0, 0, 0,
  0, 0, 0,
  0, 0, 0
};

__constant__ static float jacT[][3] =
{
  0, 0, 0,
  0, 0, 0,
  0, 0, 0,
  1, 0, 0,
  0, 1, 0,
  0, 0, 1
};

template<int numDims>
__global__ void ProjectionOntoLineAndItsJacobian(const float* pTildeP, const float* pS, const float *pT, /*const float *pJacTildeP, const float *pJacS, const float *pJacT,*/ float *pP, float* pJacP)
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

  //float jacTildeP[numDims];
  //float jacS[numDims];
  //float jacT[numDims];
  float jacP[numDims];

  const int indJac0 = numDims * numParams * numPnt;

  //for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  //{
  //  jacTildeP[i] = pJacTildeP[indJac];
  //  jacS[i] = pJacS[indJac];
  //  jacT[i] = pJacT[indJac];
  //}

  ProjectionOntoLineAndItsJacobianAt<numDims>(tildeP, s, t, jacTildeP[numPar], jacS[numPar], jacT[numPar], p, jacP);

  for (int i = 0, indPnt = indPnt0; i < numDims; ++i, ++indPnt)
  {
    pP[indPnt] = p[i];
  }

  for (int i = 0, indJac = indJac0 + numPar; i < numDims; ++i, indJac += numParams)
  {
    pJacP[indJac] = jacP[i];
  }
}

extern "C" void ProjectionOntoLineAndItsJacobian3x6(const float* pTildeP, const float* pS, const float *pT, /*const float *pJacTildeP, const float *pJacS, const float *pJacT,*/ float *pP, float* pJacP, unsigned int numPoints)
{
  ProjectionOntoLineAndItsJacobian<3><<<numPoints, 6>>>(pTildeP, pS, pT, /*pJacTildeP, pJacS, pJacT,*/ pP, pJacP);
}
