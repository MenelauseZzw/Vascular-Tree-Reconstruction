#include "Minimize.h"
#include "ProjectionOntoLine.hpp"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include "LinearCombination.hpp"
#include "LevenbergMarquardtMinimizer.hpp"

void Minimize(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP, int maxIterations)
{
  typedef cusp::device_memory MemorySpace;

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

  typedef int IndexType;
  typedef float ValueType;

  cusp::array1d<ValueType, cusp::device_memory> sAndT(numPoints * (NumDimensions + NumDimensions));

  auto s = sAndT.subarray(0, numPoints * NumDimensions);
  auto t = sAndT.subarray(numPoints * NumDimensions, numPoints * NumDimensions);

  cusp::copy(hS, s);
  cusp::copy(hT, t);

  const cusp::array1d<ValueType, cusp::device_memory> gamma(numPairs, 20);

  typedef gpuLinearCombination<3, ValueType, IndexType> CostFunctionType;

  CostFunctionType func(hTildeP, hIndPi, hIndPj, hBeta, gamma);

  // Local variables
  ValueType damp = 1000;

  const ValueType dampmin = 1e-5;
  const ValueType tolx = 1e-6;
  const ValueType tolf = 1e-6;
  const ValueType tolg = 1e-5;
  
  int itn = 0;
  LevenbergMarquardtMinimizer(func, sAndT, damp, dampmin, tolx, tolf, tolg, itn, maxIterations);

  cusp::copy(s, hS);
  cusp::copy(t, hT);

  const cusp::array1d<ValueType, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<ValueType, cusp::device_memory> p(numPoints * NumDimensions);

  gpuProjectionOntoLine<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&p[0]),
    numPoints);

  auto hP = cusp::make_array1d_view(pP, pP + NumDimensions * numPoints);

  cusp::copy(p, hP);
}

