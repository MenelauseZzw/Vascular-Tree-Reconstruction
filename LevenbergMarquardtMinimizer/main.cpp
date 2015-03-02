#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/transpose.h>

int main(int args, char *argv[])
{
  const int numDims = 3;
  const int numParams = 6;

  cusp::array1d<float, cusp::host_memory> tildeP(numDims);
  cusp::array1d<float, cusp::host_memory> s(numDims);
  cusp::array1d<float, cusp::host_memory> t(numDims);

  tildeP[0] = 0; tildeP[1] = 0; tildeP[2] = 0;
  s[0] = 1; s[1] = 1; s[2] = 1;
  t[0] = 1; t[1] = 1; t[2] = 1;

  cusp::array1d<float, cusp::host_memory> p;
  
  //projectionOntoLine(tildeP, s, t, p);

  cusp::array1d<float, cusp::host_memory> gradP(numParams);

  cusp::print(tildeP);
}

/*
#include "ProjectionOntoLineAndItsJacobian.h"
#include "PairwiseCostFunctionAndItsGradientWithRespectToParams.h"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

int main(int argc, char *argv[])
{
const size_t numPoints = 65535;
const size_t numDims = 3;
const size_t numParams = 12;

float tildePi[numDims];
float tildePj[numDims];
float si[numDims];
float ti[numDims];
float sj[numDims];
float tj[numDims];

memset(tildePi, 0, sizeof(tildePi));
memset(tildePj, 0, sizeof(tildePj));
memset(si, 0, sizeof(si));
memset(ti, 0, sizeof(ti));
memset(sj, 0, sizeof(sj));
memset(tj, 0, sizeof(tj));

tildePi[0] = 1;
tildePj[1] = 1;

sj[0] = 0.5;

ti[0] = ti[1] = ti[2] = 1;
tj[0] = tj[1] = tj[2] = 1;

float* pTildePi;
float* pTildePj;
float* pSi;
float* pTi;
float* pSj;
float* pTj;
float* pPi;
float* pPj;

const int numPntBytes = numDims * numPoints * sizeof(float);
cudaMalloc((void**)&pTildePi, numPntBytes);
cudaMalloc((void**)&pTildePj, numPntBytes);
cudaMalloc((void**)&pSi, numPntBytes);
cudaMalloc((void**)&pTi, numPntBytes);
cudaMalloc((void**)&pSj, numPntBytes);
cudaMalloc((void**)&pTj, numPntBytes);
cudaMalloc((void**)&pPi, numPntBytes);
cudaMalloc((void**)&pPj, numPntBytes);

float* pJacTildePi;
float* pJacSi;
float* pJacTi;
float* pJacPi;

const int numJacBytes = numDims * numParams * numPoints * sizeof(float);
cudaMalloc((void**)&pJacTildePi, numJacBytes);
cudaMalloc((void**)&pJacSi, numJacBytes);
cudaMalloc((void**)&pJacTi, numJacBytes);
cudaMalloc((void**)&pJacPi, numJacBytes);

float* pJacTildePj;
float* pJacSj;
float* pJacTj;
float* pJacPj;

cudaMalloc((void**)&pJacTildePj, numJacBytes);
cudaMalloc((void**)&pJacSj, numJacBytes);
cudaMalloc((void**)&pJacTj, numJacBytes);
cudaMalloc((void**)&pJacPj, numJacBytes);

float* pPairwiseCostFunctioni;
float* pPairwiseCostFunctionj;

float* pPairwiseCostGradienti;
float* pPairwiseCostGradientj;

const int numFuncBytes = numPoints * sizeof(float);
cudaMalloc((void**)&pPairwiseCostFunctioni, numFuncBytes);
cudaMalloc((void**)&pPairwiseCostFunctionj, numFuncBytes);

float* pUnaryCostGradient;

const int numGradBytes = numPoints * numParams * sizeof(float);
cudaMalloc((void**)&pPairwiseCostGradienti, numGradBytes);
cudaMalloc((void**)&pPairwiseCostGradientj, numGradBytes);

float* pTemp = new float[numPoints * numDims];

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = tildePi[0];
pTemp[i + 1] = tildePi[1];
pTemp[i + 2] = tildePi[2];
}

cudaMemcpy(pTildePi, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = tildePj[0];
pTemp[i + 1] = tildePj[1];
pTemp[i + 2] = tildePj[2];
}

cudaMemcpy(pTildePj, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = si[0];
pTemp[i + 1] = si[1];
pTemp[i + 2] = si[2];
}

cudaMemcpy(pSi, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = sj[0];
pTemp[i + 1] = sj[1];
pTemp[i + 2] = sj[2];
}

cudaMemcpy(pSj, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = ti[0];
pTemp[i + 1] = ti[1];
pTemp[i + 2] = ti[2];
}

cudaMemcpy(pTi, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = tj[0];
pTemp[i + 1] = tj[1];
pTemp[i + 2] = tj[2];
}

cudaMemcpy(pTj, pTemp, numPntBytes, cudaMemcpyHostToDevice);

delete[] pTemp;

pTemp = new float[numPoints * numDims * numParams];

memset(pTemp, 0, numJacBytes);

cudaMemcpy(pJacTildePi, pTemp, numJacBytes, cudaMemcpyHostToDevice);
cudaMemcpy(pJacTildePj, pTemp, numJacBytes, cudaMemcpyHostToDevice);

memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams) + 0 * numDims] = 1;
pTemp[i + 1 * (1 + numParams) + 0 * numDims] = 1;
pTemp[i + 2 * (1 + numParams) + 0 * numDims] = 1;
}

cudaMemcpy(pJacSi, pTemp, numJacBytes, cudaMemcpyHostToDevice);

memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams) + 1 * numDims] = 1;
pTemp[i + 1 * (1 + numParams) + 1 * numDims] = 1;
pTemp[i + 2 * (1 + numParams) + 1 * numDims] = 1;
}

cudaMemcpy(pJacTi, pTemp, numJacBytes, cudaMemcpyHostToDevice);
memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams) + 2 * numDims] = 1;
pTemp[i + 1 * (1 + numParams) + 2 * numDims] = 1;
pTemp[i + 2 * (1 + numParams) + 2 * numDims] = 1;
}

cudaMemcpy(pJacSj, pTemp, numJacBytes, cudaMemcpyHostToDevice);

memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams) + 3 * numDims] = 1;
pTemp[i + 1 * (1 + numParams) + 3 * numDims] = 1;
pTemp[i + 2 * (1 + numParams) + 3 * numDims] = 1;
}

cudaMemcpy(pJacTj, pTemp, numJacBytes, cudaMemcpyHostToDevice);

delete[] pTemp;

float jacTildeP[numDims][numParams];
float jacS[numDims][numParams];
float jacT[numDims][numParams];
float jacP[numDims][numParams];

cudaEvent_t start, stop;
float time;
//cudaEventCreate(&start);
//cudaEventCreate(&stop);
//cudaEventRecord(start, 0);

////ProjectionOntoLineAndItsJacobian3x6(pTildePi, pSi, pTi, pJacTildePi, pJacSi, pJ

//cudaEventRecord(stop, 0);
//cudaEventSynchronize(stop);
//cudaEventElapsedTime(&time, start, stop);
//std::cout << "Time for the kernel <ProjectionOntoLineAndItsJacobian3x6>: " << time << " ms" << std::endl;

//float p[numDims];

//cudaMemcpy(p, &pPi[numDims * (numPoints - 1)], sizeof(p), cudaMemcpyDeviceToHost);
//cudaMemcpy(jacP, &pJacPi[numDims * numParams * (numPoints - 1)], sizeof(jacP), cudaMemcpyDeviceToHost);

//std::cout << "p" << std::endl;
//for (int numDim = 0; numDim < numDims; ++numDim)
//{
//  std::cout << p[numDim] << std::endl;
//}

//std::cout << "jacP" << std::endl;
//for (int numDim = 0; numDim < numDims; ++numDim)
//{
//  for (int numParam = 0; numParam < numParams; ++numParam)
//  {
//    std::cout << jacP[numDim][numParam] << " ";
//  }
//  std::cout << std::endl;
//}

//cudaEventDestroy(start);
//cudaEventDestroy(stop);

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);

PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(pTildePi, pSi, pTi, pJacTildePi, pJacSi, pJacTi, pTildePj, pSj, pTj, pJacTildePj, pJacSj, pJacTj, pPairwiseCostFunctioni, pPairwiseCostFunctionj, pPairwiseCostGradienti, pPairwiseCostGradientj, numPoints);

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
std::cout << std::endl << "Time for the kernel <PairwiseCostFunctionAndItsGradientWithRespectToParams3x12>: " << time << " ms" << std::endl;

float pairwiseCostFunction;
cudaMemcpy(&pairwiseCostFunction, &pPairwiseCostFunctioni[numPoints - 1], sizeof(pairwiseCostFunction), cudaMemcpyDeviceToHost);

float pairwiseCostGradient[numParams];
cudaMemcpy(pairwiseCostGradient, &pPairwiseCostGradienti[numParams * (numPoints - 1)], sizeof(pairwiseCostGradient), cudaMemcpyDeviceToHost);

std::cout << "Pairwise cost function: " << pairwiseCostFunction << std::endl;

std::cout << "Pairwise cost gradient" << std::endl;
for (int numParam = 0; numParam < numParams; ++numParam)
{
std::cout << pairwiseCostGradient[numParam] << std::endl;
}

cudaFree(pPairwiseCostGradientj);
cudaFree(pPairwiseCostGradienti);
cudaFree(pPairwiseCostFunctionj);
cudaFree(pPairwiseCostFunctioni);
cudaFree(pJacPj);
cudaFree(pJacPi);
cudaFree(pJacTj);
cudaFree(pJacTi);
cudaFree(pJacSj);
cudaFree(pJacSi);
cudaFree(pJacTildePj);
cudaFree(pJacTildePi);
cudaFree(pPj);
cudaFree(pPi);
cudaFree(pTj);
cudaFree(pSj);
cudaFree(pTi);
cudaFree(pSi);
cudaFree(pTildePj);
cudaFree(pTildePi);
}*/

/*int main(int argc, char *argv[])
{
const size_t numPoints = 65535;
const size_t numDims = 3;
const size_t numParams = 6;

float tildeP[numDims];
float s[numDims];
float t[numDims];

memset(tildeP, 0, sizeof(tildeP));
memset(s, 0, sizeof(s));
memset(t, 0, sizeof(t));

tildeP[0] = 1;
t[0] = t[1] = t[2] = 1;

float* pTildeP;
float* pS;
float* pT;
float* pP;

const int numPntBytes = numDims * numPoints * sizeof(float);
cudaMalloc((void**)&pTildeP, numPntBytes);
cudaMalloc((void**)&pS, numPntBytes);
cudaMalloc((void**)&pT, numPntBytes);
cudaMalloc((void**)&pP, numPntBytes);

float* pJacTildeP;
float* pJacS;
float* pJacT;
float* pJacP;

const int numJacBytes = numDims * numParams * numPoints * sizeof(float);
cudaMalloc((void**)&pJacTildeP, numJacBytes);
cudaMalloc((void**)&pJacS, numJacBytes);
cudaMalloc((void**)&pJacT, numJacBytes);
cudaMalloc((void**)&pJacP, numJacBytes);

float* pUnaryCostFunction;

const int numFuncBytes = numPoints * sizeof(float);
cudaMalloc((void**)&pUnaryCostFunction, numFuncBytes);

float* pUnaryCostGradient;

const int numGradBytes = numPoints * numParams * sizeof(float);
cudaMalloc((void**)&pUnaryCostGradient, numGradBytes);

float* pTemp = new float[numPoints * numDims];

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = tildeP[0];
pTemp[i + 1] = tildeP[1];
pTemp[i + 2] = tildeP[2];
}

cudaMemcpy(pTildeP, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = s[0];
pTemp[i + 1] = s[1];
pTemp[i + 2] = s[2];
}

cudaMemcpy(pS, pTemp, numPntBytes, cudaMemcpyHostToDevice);

for (int i = 0; i < numPoints * numDims; i += numDims)
{
pTemp[i + 0] = t[0];
pTemp[i + 1] = t[1];
pTemp[i + 2] = t[2];
}

cudaMemcpy(pT, pTemp, numPntBytes, cudaMemcpyHostToDevice);

delete[] pTemp;

pTemp = new float[numPoints * numDims * numParams];

memset(pTemp, 0, numJacBytes);

cudaMemcpy(pJacTildeP, pTemp, numJacBytes, cudaMemcpyHostToDevice);

memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams)] = 1;
pTemp[i + 1 * (1 + numParams)] = 1;
pTemp[i + 2 * (1 + numParams)] = 1;
}

cudaMemcpy(pJacS, pTemp, numJacBytes, cudaMemcpyHostToDevice);

memset(pTemp, 0, numJacBytes);

for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
{
pTemp[i + 0 * (1 + numParams) + numDims] = 1;
pTemp[i + 1 * (1 + numParams) + numDims] = 1;
pTemp[i + 2 * (1 + numParams) + numDims] = 1;
}

cudaMemcpy(pJacT, pTemp, numJacBytes, cudaMemcpyHostToDevice);

delete[] pTemp;

float jacTildeP[numDims][numParams];
float jacS[numDims][numParams];
float jacT[numDims][numParams];
float jacP[numDims][numParams];

cudaEvent_t start, stop;
float time;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

ProjectionOntoLineAndItsJacobian3x6(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pP, pJacP, numPoints);

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
std::cout << "Time for the kernel <ProjectionOntoLineAndItsJacobian3x6>: " << time << " ms" << std::endl;

float p[numDims];

cudaMemcpy(p, &pP[numDims * (numPoints - 1)], sizeof(p), cudaMemcpyDeviceToHost);
cudaMemcpy(jacP, &pJacP[numDims * numParams * (numPoints - 1)], sizeof(jacP), cudaMemcpyDeviceToHost);

std::cout << "p" << std::endl;
for (int numDim = 0; numDim < numDims; ++numDim)
{
std::cout << p[numDim] << std::endl;
}

std::cout << "jacP" << std::endl;
for (int numDim = 0; numDim < numDims; ++numDim)
{
for (int numParam = 0; numParam < numParams; ++numParam)
{
std::cout << jacP[numDim][numParam] << " ";
}
std::cout << std::endl;
}

cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);

UnaryCostFunctionAndItsGradientWithRespectToParams3x6(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pUnaryCostFunction, pUnaryCostGradient, numPoints);

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time, start, stop);
std::cout << std::endl << "Time for the kernel <UnaryCostFunctionAndItsGradientWithRespectToParams3x6>: " << time << " ms" << std::endl;

float unaryCostFunction;
cudaMemcpy(&unaryCostFunction, &pUnaryCostFunction[numPoints - 1], sizeof(unaryCostFunction), cudaMemcpyDeviceToHost);

float unaryCostGradient[numParams];
cudaMemcpy(unaryCostGradient, &pUnaryCostGradient[numParams * (numPoints - 1)], sizeof(unaryCostGradient), cudaMemcpyDeviceToHost);

std::cout << "Unary cost function: " << unaryCostFunction << std::endl;

std::cout << "Unary cost gradient" << std::endl;
for (int numParam = 0; numParam < numParams; ++numParam)
{
std::cout << unaryCostGradient[numParam] << std::endl;
}

cudaFree(pUnaryCostGradient);
cudaFree(pUnaryCostFunction);
cudaFree(pJacP);
cudaFree(pJacT);
cudaFree(pJacS);
cudaFree(pJacTildeP);
cudaFree(pP);
cudaFree(pT);
cudaFree(pS);
cudaFree(pTildeP);
}*/

//#include "PairwiseCostFunction.h"
//#include "UnaryCostFunction.h"
//#include <iostream>
//#include <H5Cpp.h>
//#include <thrust/device_vector.h>
//#include <thrust/iterator/zip_iterator.h>
//#include <thrust/host_vector.h>
//#include <thrust/transform_reduce.h>
//
//#ifndef H5_NO_NAMESPACE
//
//using H5::H5File;
//using H5::DataSet;
//using H5::DataSpace;
//using H5::PredType;
//
//#endif
//
//thrust::host_vector<float> createVector(H5File sourceFile, const H5std_string& targetName)
//{
//  DataSet targetDataSet = sourceFile.openDataSet(targetName);
//  DataSpace targetSpace = targetDataSet.getSpace();
//  hssize_t numPoints = targetSpace.getSimpleExtentNpoints();
//
//  thrust::host_vector<float> targetVector(numPoints);
//  targetDataSet.read(&targetVector[0], PredType::NATIVE_FLOAT);
//  return targetVector;
//}

//int main(int argc, char *argv[])
//{
//  H5File sourceFile("D:\\WesternU\\test.h5", H5F_ACC_RDONLY);
//
//  thrust::host_vector<float> px{ createVector(sourceFile, "~p.x") };
//  thrust::host_vector<float> py{ createVector(sourceFile, "~p.y") };
//  thrust::host_vector<float> pz{ createVector(sourceFile, "~p.z") };
//
//  thrust::host_vector<float> sx{ createVector(sourceFile, "s.x") };
//  thrust::host_vector<float> sy{ createVector(sourceFile, "s.y") };
//  thrust::host_vector<float> sz{ createVector(sourceFile, "s.z") };
//
//  thrust::host_vector<float> tx{ createVector(sourceFile, "t.x") };
//  thrust::host_vector<float> ty{ createVector(sourceFile, "t.y") };
//  thrust::host_vector<float> tz{ createVector(sourceFile, "t.z") };
//
//  thrust::host_vector<float> sigma{ createVector(sourceFile, "sigma") };
//
//  auto numPoints = px.size();
//  thrust::host_vector<float> unaryCostFunctions(numPoints);
//
//  sourceFile.close();
//
//  //thrust::transform(
//  //  thrust::make_zip_iterator(thrust::make_tuple(px.begin(), py.begin(), pz.begin())),
//  //  thrust::make_zip_iterator(thrust::make_tuple(px.end(), py.end(), pz.end())),
//  //  thrust::make_zip_iterator(thrust::make_tuple(sx.begin(), sy.begin(), sz.begin(), tx.begin(), ty.begin(), tz.begin(), sigma.begin())),
//  //  unaryCostFunctions.begin(),
//  //  UnaryCostFunction()
//  //  );
//
//  //for (int i = 0; i < unaryCostFunctions.size(); ++i)
//  //{
//  //  std::cout << unaryCostFunctions[i] << std::endl;
//  //}
//
//  float unaryCost = thrust::reduce(unaryCostFunctions.begin(), unaryCostFunctions.end());
//  std::cout << unaryCost << std::endl;
//
//  auto f = UnaryCostFunction();
//  auto grad = UnaryCostFunction::GradientWithRespectToParams();
//
//}