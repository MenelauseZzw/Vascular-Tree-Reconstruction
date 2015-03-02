#include "ProjectionOntoLineAndItsJacobian.h"
#include "TestProjectionOntoLineAndUnaryCostFunction.h"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/print.h>
#include <iostream>
#include <thrust/device_ptr.h>

void testProjectionOntoLineAndUnaryCostFunction(size_t numPoints, size_t maxPoints/*= 65535*/)
{
  const size_t numDims = 3;
  const size_t numParams = 6;

  typedef cusp::array1d<float, cusp::host_memory> HostArray1d;
  typedef cusp::array2d<float, cusp::host_memory, cusp::row_major> HostArray2d;

  HostArray2d tildeP(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    tildeP(i, 0) = 1;
    tildeP(i, 1) = 0;
    tildeP(i, 2) = 0;
  }

  HostArray2d s(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    s(i, 0) = 0;
    s(i, 1) = 0;
    s(i, 2) = 0;
  }

  HostArray2d t(numPoints, numDims);

  for (size_t i = 0; i < numPoints; ++i)
  {
    t(i, 0) = 1;
    t(i, 1) = 1;
    t(i, 2) = 1;
  }

  typedef cusp::array1d<float, cusp::device_memory> DeviceArray1d;
  typedef cusp::array2d<float, cusp::device_memory, cusp::row_major> DeviceArray2d;

  DeviceArray2d deviceTildeP(tildeP);
  DeviceArray2d deviceS(s);
  DeviceArray2d deviceT(t);
  DeviceArray2d deviceP(numPoints, numDims);

  HostArray2d jacTildeP(numPoints, numDims * numParams, 0);

  HostArray2d jacS(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacS.row(i));

    jac(0, 0) = 1;
    jac(1, 1) = 1;
    jac(2, 2) = 1;
  }

  HostArray2d jacT(numPoints, numDims * numParams, 0);

  for (size_t i = 0; i < numPoints; ++i)
  {
    cusp::array2d_view<HostArray2d::row_view, cusp::row_major> jac(numDims, numParams, numParams, jacT.row(i));

    jac(0, 3) = 1;
    jac(1, 4) = 1;
    jac(2, 5) = 1;
  }

  DeviceArray2d deviceJacTildeP(jacTildeP);
  DeviceArray2d deviceJacS(jacS);
  DeviceArray2d deviceJacT(jacT);
  DeviceArray2d deviceJacP(numPoints, numDims * numParams);

  cudaEvent_t start;
  cudaEventCreate(&start);

  cudaEvent_t stop;
  cudaEventCreate(&stop);
  
  cudaEventRecord(start, 0);

  for (int i = 0; i < numPoints; i += maxPoints)
  {
    float* pTildeP = thrust::raw_pointer_cast(&deviceTildeP(i, 0));
    float* pS = thrust::raw_pointer_cast(&deviceS(i, 0));
    float* pT = thrust::raw_pointer_cast(&deviceT(i, 0));
    float* pP = thrust::raw_pointer_cast(&deviceP(i, 0));

    float* pJacTildeP = thrust::raw_pointer_cast(&deviceJacTildeP(i, 0));
    float* pJacS = thrust::raw_pointer_cast(&deviceJacS(i, 0));
    float* pJacT = thrust::raw_pointer_cast(&deviceJacT(i, 0));
    float* pJacP = thrust::raw_pointer_cast(&deviceJacP(i, 0));

    ProjectionOntoLineAndItsJacobian3x6(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pP, pJacP, std::min(numPoints - i, maxPoints));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float timeElapsedMs;
  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for the kernel <ProjectionOntoLineAndItsJacobian3x6> " << timeElapsedMs << " ms" << std::endl;

  DeviceArray1d e(numPoints);

  const size_t numCols = numPoints * numParams;
  HostArray1d columnIndices(numCols);
  for (int i = 0; i < numCols; ++i)
  {
    columnIndices[i] = i;
  }

  HostArray1d rowOffsets(numPoints + 1);
  for (int i = 0; i <= numPoints; ++i)
  {
    rowOffsets[i] = i * numParams;
  }

  typedef cusp::csr_matrix<int, float, cusp::device_memory> DeviceCsrMatrix;
  DeviceCsrMatrix jacE(numPoints, numCols, numPoints * numParams);
  jacE.row_offsets = rowOffsets;
  jacE.column_indices = columnIndices;

  cudaEventRecord(start, 0);

  for (int i = 0; i < numPoints; i += maxPoints)
  {
    float* pTildeP = thrust::raw_pointer_cast(&deviceTildeP(i, 0));
    float* pS = thrust::raw_pointer_cast(&deviceS(i, 0));
    float* pT = thrust::raw_pointer_cast(&deviceT(i, 0));
    float* pP = thrust::raw_pointer_cast(&deviceP(i, 0));

    float* pJacTildeP = thrust::raw_pointer_cast(&deviceJacTildeP(i, 0));
    float* pJacS = thrust::raw_pointer_cast(&deviceJacS(i, 0));
    float* pJacT = thrust::raw_pointer_cast(&deviceJacT(i, 0));
    float* pJacP = thrust::raw_pointer_cast(&deviceJacP(i, 0));

    float* pE = thrust::raw_pointer_cast(&e[i]);
    float* pJacE = thrust::raw_pointer_cast(&jacE.values[i * numParams]);

    UnaryCostFunctionAndItsGradientWithRespectToParams3x6(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pE, pJacE, std::min(numPoints - i, maxPoints));
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for the kernel <UnaryCostFunctionAndItsGradientWithRespectToParams3x6> " << timeElapsedMs << " ms" << std::endl;

  HostArray1d p(deviceP.row(numPoints - 1));
  std::cout << "p" << std::endl;
  cusp::print(p);

  HostArray2d jacP(
    cusp::array2d_view<DeviceArray2d::row_view, cusp::row_major>(numDims, numParams, numParams, deviceJacP.row(numPoints - 1))
    );

  std::cout << "jacP" << std::endl;
  cusp::print(jacP);

  //std::cout << "e" << std::endl;
  //cusp::print(e);

  //std::cout << "jacE" << std::endl;
  //cusp::print(jacE);

  DeviceCsrMatrix jacEt;

  cudaEventRecord(start, 0);

  cusp::transpose(jacE, jacEt);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for transpose(jacE) " << timeElapsedMs << " ms" << std::endl;

  DeviceCsrMatrix jacEtTimesJacE;

  cudaEventRecord(start, 0);

  cusp::multiply(jacEt, jacE, jacEtTimesJacE);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for multiply(jacEt, jacE) " << timeElapsedMs << " ms" << std::endl;
  
  DeviceArray1d jacEtTimesE(jacEt.num_rows);

  cudaEventRecord(start, 0);
  cusp::multiply(jacEt, e, jacEtTimesE);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&timeElapsedMs, start, stop);
  std::cout << "Time for multiply(jacEt, e) " << timeElapsedMs << " ms" << std::endl;
}
