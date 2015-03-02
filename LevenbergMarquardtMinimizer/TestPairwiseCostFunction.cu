#include "PairwiseCostFunctionAndItsGradientWithRespectToParams.h"
#include "TestPairwiseCostFunction.h"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <iostream>
#include <thrust/device_ptr.h>

void testPairwiseCostFunction(size_t numPoints, size_t maxPoints/*= 65535*/)
{
  const size_t numDims = 3;
  const size_t numParams = 12;

  typedef cusp::array1d<float, cusp::host_memory> HostArray1d;
  typedef cusp::array2d<float, cusp::host_memory, cusp::row_major> HostArray2d;

  HostArray2d tildePi(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    tildePi(i, 0) = 1;
    tildePi(i, 1) = 0;
    tildePi(i, 2) = 0;
  }

  HostArray2d si(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    si(i, 0) = 0;
    si(i, 1) = 0;
    si(i, 2) = 0;
  }

  HostArray2d ti(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    ti(i, 0) = 1;
    ti(i, 1) = 1;
    ti(i, 2) = 1;
  }

  HostArray2d tildePj(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    tildePj(i, 0) = 0;
    tildePj(i, 1) = 1;
    tildePj(i, 2) = 0;
  }

  HostArray2d sj(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    sj(i, 0) = 0.5;
    sj(i, 1) = 0;
    sj(i, 2) = 0;
  }

  HostArray2d tj(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    tj(i, 0) = 1;
    tj(i, 1) = 1;
    tj(i, 2) = 1;
  }

  typedef cusp::array1d<float, cusp::device_memory> DeviceArray1d;
  typedef cusp::array2d<float, cusp::device_memory, cusp::row_major> DeviceArray2d;

  DeviceArray2d deviceTildePi(tildePi);
  DeviceArray2d deviceSi(si);
  DeviceArray2d deviceTi(ti);

  DeviceArray2d deviceTildePj(tildePj);
  DeviceArray2d deviceSj(sj);
  DeviceArray2d deviceTj(tj);

  HostArray2d jacTildePi(numPoints, numDims * numParams, 0);

  HostArray2d jacSi(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacSi.row(i));

    jac(0, 0) = 1;
    jac(1, 1) = 1;
    jac(2, 2) = 1;
  }

  HostArray2d jacTi(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacTi.row(i));

    jac(0, 3) = 1;
    jac(1, 4) = 1;
    jac(2, 5) = 1;
  }

  HostArray2d jacTildePj(numPoints, numDims * numParams, 0);

  HostArray2d jacSj(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacSj.row(i));

    jac(0, 6) = 1;
    jac(1, 7) = 1;
    jac(2, 8) = 1;
  }

  HostArray2d jacTj(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacTj.row(i));

    jac(0, 9) = 1;
    jac(1,10) = 1;
    jac(2,11) = 1;
  }

  DeviceArray2d deviceJacTildePi(jacTildePi);
  DeviceArray2d deviceJacSi(jacSi);
  DeviceArray2d deviceJacTi(jacTi);

  DeviceArray2d deviceJacTildePj(jacTildePj);
  DeviceArray2d deviceJacSj(jacSj);
  DeviceArray2d deviceJacTj(jacTj);

  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);

  DeviceArray1d devicePairwiseCostFunctioni(numPoints);
  DeviceArray2d devicePairwiseCostGradienti(numPoints, numParams);

  DeviceArray1d devicePairwiseCostFunctionj(numPoints);
  DeviceArray2d devicePairwiseCostGradientj(numPoints, numParams);

  cudaEventRecord(start, 0);

  for (int i = 0; i < numPoints; i += maxPoints)
  {
    float* pTildePi = thrust::raw_pointer_cast(&deviceTildePi(i, 0));
    float* pSi = thrust::raw_pointer_cast(&deviceSi(i, 0));
    float* pTi = thrust::raw_pointer_cast(&deviceTi(i, 0));

    float* pTildePj = thrust::raw_pointer_cast(&deviceTildePj(i, 0));
    float* pSj = thrust::raw_pointer_cast(&deviceSj(i, 0));
    float* pTj = thrust::raw_pointer_cast(&deviceTj(i, 0));

    float* pJacTildePi = thrust::raw_pointer_cast(&deviceJacTildePi(i, 0));
    float* pJacSi = thrust::raw_pointer_cast(&deviceJacSi(i, 0));
    float* pJacTi = thrust::raw_pointer_cast(&deviceJacTi(i, 0));

    float* pJacTildePj = thrust::raw_pointer_cast(&deviceJacTildePj(i, 0));
    float* pJacSj = thrust::raw_pointer_cast(&deviceJacSj(i, 0));
    float* pJacTj = thrust::raw_pointer_cast(&deviceJacTj(i, 0));

    float* pPairwiseCostFunctioni = thrust::raw_pointer_cast(&devicePairwiseCostFunctioni[i]);
    float* pPairwiseCostGradienti = thrust::raw_pointer_cast(&devicePairwiseCostGradienti(i, 0));

    float* pPairwiseCostFunctionj = thrust::raw_pointer_cast(&devicePairwiseCostFunctionj[i]);
    float* pPairwiseCostGradientj = thrust::raw_pointer_cast(&devicePairwiseCostGradientj(i, 0));

    PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(
      pTildePi, pSi, pTi, pJacTildePi, pJacSi, pJacTi,
      pTildePj, pSj, pTj, pJacTildePj, pJacSj, pJacTj,
      pPairwiseCostFunctioni, pPairwiseCostFunctionj,
      pPairwiseCostGradienti, pPairwiseCostGradientj, std::min(numPoints - i, maxPoints));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float timeElapsedMs;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for the kernel <PairwiseCostFunctionAndItsGradientWithRespectToParams3x12> " << timeElapsedMs << " ms" << std::endl;

  float pairwiseCostFunctioni = devicePairwiseCostFunctioni[numPoints - 1];
  std::cout << "Pairwise cost function (i) " << pairwiseCostFunctioni << std::endl;

  HostArray1d pairwiseCostGradienti = devicePairwiseCostGradienti.row(numPoints - 1);
  std::cout << "Pairwise cost gradient (i) " << std::endl;
  cusp::print(pairwiseCostGradienti);

  float pairwiseCostFunctionj = devicePairwiseCostFunctionj[numPoints - 1];
  std::cout << "Pairwise cost function (j) " << pairwiseCostFunctionj << std::endl;

  HostArray1d pairwiseCostGradientj = devicePairwiseCostGradientj.row(numPoints - 1);
  std::cout << "Pairwise cost gradient (j) " << std::endl;
  cusp::print(pairwiseCostGradientj);
}