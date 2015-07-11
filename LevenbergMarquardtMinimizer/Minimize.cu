#include "Minimize.hpp"
#include "ProjectionOntoLine.hpp"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include "LinearCombination.hpp"
#include "LevenbergMarquardtMinimizer.hpp"

template<typename ValueType, typename IndexType>
void Minimize(const ValueType* pMeasurements, ValueType* pTangentLinesPoints1, ValueType* pTangentLinesPoints2, const ValueType* pRadiuses, int numMeasurements, const IndexType* pIndices1, const IndexType* pIndices2, int indicesLength, ValueType* pPositions, int itnlim, ValueType* pDistances, ValueType* pCurvatures)
{
  typedef cusp::device_memory MemorySpace;

  // Local constants
  const int NumDimensions = 3;

  auto hTildeP = cusp::make_array1d_view(pMeasurements, pMeasurements + NumDimensions * numMeasurements);
  auto hS = cusp::make_array1d_view(pTangentLinesPoints1, pTangentLinesPoints1 + NumDimensions * numMeasurements);
  auto hT = cusp::make_array1d_view(pTangentLinesPoints2, pTangentLinesPoints2 + NumDimensions * numMeasurements);
  auto hSigma = cusp::make_array1d_view(pRadiuses, pRadiuses + numMeasurements);

  auto hIndPi = cusp::make_array1d_view(pIndices1, pIndices1 + indicesLength);
  auto hIndPj = cusp::make_array1d_view(pIndices2, pIndices2 + indicesLength);

  cusp::array1d<ValueType, cusp::host_memory> hBeta(numMeasurements);

  for (int i = 0; i < hBeta.size(); ++i)
  {
    hBeta[i] = 1 / hSigma[i];
  }

  cusp::array1d<ValueType, cusp::device_memory> sAndT(numMeasurements * (NumDimensions + NumDimensions));

  auto s = sAndT.subarray(0, numMeasurements * NumDimensions);
  auto t = sAndT.subarray(numMeasurements * NumDimensions, numMeasurements * NumDimensions);

  cusp::copy(hS, s);
  cusp::copy(hT, t);

  const cusp::array1d<ValueType, cusp::device_memory> gamma(indicesLength, 20);

  typedef gpuLinearCombination<3, ValueType, IndexType> CostFunctionType;

  CostFunctionType func(hTildeP, hIndPi, hIndPj, hBeta, gamma);

  // Local variables
  ValueType damp = 1000;

  const ValueType dampmin = 1e-5;
  const ValueType tolx = 1e-6;
  const ValueType tolf = 1e-6;
  const ValueType tolg = 1e-5;
  
  int itn = 0;
  LevenbergMarquardtMinimizer(func, sAndT, damp, dampmin, tolx, tolf, tolg, itn, itnlim);

  cusp::copy(s, hS);
  cusp::copy(t, hT);

  const cusp::array1d<ValueType, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<ValueType, cusp::device_memory> p(numMeasurements * NumDimensions);

  gpuProjectionOntoLine<ValueType, NumDimensions>(
    thrust::raw_pointer_cast(&tildeP[0]),
    thrust::raw_pointer_cast(&s[0]),
    thrust::raw_pointer_cast(&t[0]),
    thrust::raw_pointer_cast(&p[0]),
    numMeasurements);
  
  const cusp::array1d<ValueType, cusp::host_memory> hWeights1(numMeasurements / NumDimensions, 1);
  const cusp::array1d<ValueType, cusp::host_memory> hWeights2(indicesLength, 1);

  cpuDistanceCostResidual<ValueType, 3>(&hTildeP[0], &hS[0], &hT[0], &hWeights1[0], pDistances, numMeasurements / NumDimensions);
  cpuCurvatureCostResidual<ValueType, IndexType, 3>(&hTildeP[0], &hS[0], &hT[0], &hWeights2[0], &hIndPi[0], &hIndPj[0], pCurvatures, indicesLength);

  auto hP = cusp::make_array1d_view(pPositions, pPositions + NumDimensions * numMeasurements);
  cusp::copy(p, hP);
}

//Explicit instantiation
#define InstantiateMinimize(ValueType, IndexType) \
template void Minimize(const ValueType*, ValueType*, ValueType*, const ValueType*, int, const IndexType*, const IndexType*, int, ValueType*, int, ValueType*, ValueType*)

InstantiateMinimize(float, int);
InstantiateMinimize(double, int);