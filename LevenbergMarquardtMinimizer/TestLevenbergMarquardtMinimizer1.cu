#include "TestLevenbergMarquardtMinimizer1.h"
#include "ProjectionOntoLine.hpp"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include "CostFunction.hpp"
#include "LevenbergMarquardtMinimizer.hpp"

void testLevenbergMarquardtMinimizer1(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP, int maxIterations)
{
  typedef float ValueType;
  typedef cusp::device_memory MemorySpace;

  // Local variables
  float damp = 1000;
  const float dampmin = 1e-5;

  // Local constants
  const int numDimensions = 3;

  auto hTildeP = cusp::make_array1d_view(pTildeP, pTildeP + numDimensions * numPoints);
  auto hS = cusp::make_array1d_view(pS, pS + numDimensions * numPoints);
  auto hT = cusp::make_array1d_view(pT, pT + numDimensions * numPoints);
  auto hSigma = cusp::make_array1d_view(pSigma, pSigma + numPoints);

  auto hIndPi = cusp::make_array1d_view(pIndPi, pIndPi + numPairs);
  auto hIndPj = cusp::make_array1d_view(pIndPj, pIndPj + numPairs);

  cusp::array1d<float, cusp::host_memory> hBeta(numPoints);

  for (int i = 0; i < hBeta.size(); ++i)
  {
    hBeta[i] = 1 / hSigma[i];
  }

  cusp::array1d<float, cusp::device_memory> sAndT(numPoints * (numDimensions + numDimensions));

  auto s = sAndT.subarray(0, numPoints * numDimensions);
  auto t = sAndT.subarray(numPoints * numDimensions, numPoints * numDimensions);

  cusp::copy(hS, s);
  cusp::copy(hT, t);

  const cusp::array1d<float, cusp::device_memory> gamma(numPairs, 20);

  typedef CostFunction<3, int, ValueType, cusp::device_memory> CostFunctionType;

  CostFunctionType costFunction(hTildeP, hBeta, gamma, hIndPi, hIndPj);

  auto allocateJacobian = [&costFunction]()
  {
    return costFunction.AllocateJacobian();
  };

  auto computeJacobian = [&costFunction](const cusp::array1d<float, cusp::device_memory>& st, cusp::array1d<float, cusp::device_memory>& jacobian)
  {
    return costFunction.ComputeJacobian(st, jacobian);
  };

  auto computeResidual = [&](const cusp::array1d<float, cusp::device_memory>& st, cusp::array1d<ValueType, MemorySpace>& residual)
  {
    return costFunction.ComputeResidual(st, residual);
  };

  LevenbergMarquardtMinimizer(allocateJacobian, computeJacobian, computeResidual, sAndT, damp, dampmin);

  cusp::copy(s, hS);
  cusp::copy(t, hT);

  const cusp::array1d<float, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<float, cusp::device_memory> p(numPoints * numDimensions);

  ProjectionOntoLine<float, 3>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&p[0]),
    numPoints
    );

  auto hP = cusp::make_array1d_view(pP, pP + numDimensions * numPoints);

  cusp::copy(p, hP);
}
