#include "TestLevenbergMarquardtMinimizer1.h"
#include "ProjectionOntoLine.hpp"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include "LinearCombination.hpp"
#include "LevenbergMarquardtMinimizer.hpp"

void testLevenbergMarquardtMinimizer1(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP, int maxIterations)
{
  typedef int IndexType;
  typedef float ValueType;
  typedef cusp::device_memory MemorySpace;

  // Local variables
  float damp = 1000;
  const float dampmin = 1e-5;

  // Local constants
  const int NumDimensions = 3;

  auto hTildeP = cusp::make_array1d_view(pTildeP, pTildeP + NumDimensions * numPoints);
  auto hS = cusp::make_array1d_view(pS, pS + NumDimensions * numPoints);
  auto hT = cusp::make_array1d_view(pT, pT + NumDimensions * numPoints);
  auto hSigma = cusp::make_array1d_view(pSigma, pSigma + numPoints);

  auto hIndPi = cusp::make_array1d_view(pIndPi, pIndPi + numPairs);
  auto hIndPj = cusp::make_array1d_view(pIndPj, pIndPj + numPairs);

  cusp::array1d<float, cusp::host_memory> hBeta(numPoints);

  for (int i = 0; i < hBeta.size(); ++i)
  {
    hBeta[i] = 1 / hSigma[i];
  }

  cusp::array1d<float, cusp::device_memory> sAndT(numPoints * (NumDimensions + NumDimensions));

  auto s = sAndT.subarray(0, numPoints * NumDimensions);
  auto t = sAndT.subarray(numPoints * NumDimensions, numPoints * NumDimensions);

  cusp::copy(hS, s);
  cusp::copy(hT, t);

  const cusp::array1d<float, cusp::device_memory> gamma(numPairs, 20);

  typedef gpuLinearCombination<3, float> CostFunctionType;

  CostFunctionType func(hTildeP, hIndPi, hIndPj, hBeta, gamma);

  LevenbergMarquardtMinimizer(func, sAndT, damp, dampmin);

  cusp::copy(s, hS);
  cusp::copy(t, hT);

  const cusp::array1d<float, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<float, cusp::device_memory> p(numPoints * NumDimensions);

  ProjectionOntoLine<float, 3>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&p[0]),
    numPoints);

  auto hP = cusp::make_array1d_view(pP, pP + NumDimensions * numPoints);

  cusp::copy(p, hP);
}
