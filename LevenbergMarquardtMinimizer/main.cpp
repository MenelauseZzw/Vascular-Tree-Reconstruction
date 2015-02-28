#include "ProjectionOntoLineAndItsJacobian.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

int main(int argc, char *argv[])
{
  const size_t numPoints = 50000;
  const size_t numDims = 3;
  const size_t numParams = 6;

  float* pTildeP;
  float* pS;
  float* pT;
  float* pP;
  float* pJacTildeP;
  float* pJacS;
  float* pJacT;
  float* pJacP;

  const unsigned int numPntBytes = numDims * numPoints * sizeof(float);
  cudaMalloc((void**)&pTildeP, numPntBytes);
  cudaMalloc((void**)&pS, numPntBytes);
  cudaMalloc((void**)&pT, numPntBytes);
  cudaMalloc((void**)&pP, numPntBytes);

  const unsigned int numJacBytes = numDims * numParams * numPoints * sizeof(float);
  cudaMalloc((void**)&pJacTildeP, numJacBytes);
  cudaMalloc((void**)&pJacS, numJacBytes);
  cudaMalloc((void**)&pJacT, numJacBytes);
  cudaMalloc((void**)&pJacP, numJacBytes);

  float* pTemp = new float[numPoints * numDims];
  for (int i = 0; i < numPoints * numDims; i += numDims)
  {
    pTemp[i + 0] = 0;
    pTemp[i + 1] = 0;
    pTemp[i + 2] = 0;
  }

  cudaMemcpy(pTildeP, pTemp, numPntBytes, cudaMemcpyHostToDevice);

  for (int i = 0; i < numPoints * numDims; i += numDims)
  {
    pTemp[i + 0] = 1;
    pTemp[i + 1] = 0;
    pTemp[i + 2] = 0;
  }

  cudaMemcpy(pS, pTemp, numPntBytes, cudaMemcpyHostToDevice);

  for (int i = 0; i < numPoints * numDims; i += numDims)
  {
    pTemp[i + 0] = 0;
    pTemp[i + 1] = 1;
    pTemp[i + 2] = 0;
  }

  cudaMemcpy(pT, pTemp, numPntBytes, cudaMemcpyHostToDevice);

  delete[] pTemp;

  pTemp = new float[numPoints * numDims * numParams];
  for (int i = 0; i < numPoints * numDims * numParams; ++i)
  {
    pTemp[i] = 0;
  }

  cudaMemcpy(pJacTildeP, pTemp, numJacBytes, cudaMemcpyHostToDevice);

  for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
  {
    pTemp[i + 0 * (1 + numParams)] = 1;
    pTemp[i + 1 * (1 + numParams)] = 1;
    pTemp[i + 2 * (1 + numParams)] = 1;
  }

  cudaMemcpy(pJacS, pTemp, numJacBytes, cudaMemcpyHostToDevice);

  for (int i = 0; i < numPoints * numDims * numParams; ++i)
  {
    pTemp[i] = 0;
  }

  for (int i = 0; i < numPoints * numDims * numParams; i += numDims * numParams)
  {
    pTemp[i + 0 * (1 + numParams) + numDims] = 1;
    pTemp[i + 1 * (1 + numParams) + numDims] = 1;
    pTemp[i + 2 * (1 + numParams) + numDims] = 1;
  }

  cudaMemcpy(pJacT, pTemp, numJacBytes, cudaMemcpyHostToDevice);
  
  delete[] pTemp;

  float tildeP[numDims];
  float s[numDims];
  float t[numDims];
  float p[numDims];
  /*
  std::memset(tildeP, 0, sizeof(tildeP));
  std::memset(s, 0, sizeof(s));
  std::memset(t, 0, sizeof(t));
  std::memset(p, 0, sizeof(p));

  s[0] = 1;
  t[1] = 1;
  */

  float jacTildeP[numDims][numParams];
  float jacS[numDims][numParams];
  float jacT[numDims][numParams];
  float jacP[numDims][numParams];
  /*
  std::memset(jacTildeP, 0, sizeof(jacTildeP));
  std::memset(jacS, 0, sizeof(jacS));
  std::memset(jacT, 0, sizeof(jacT));
  std::memset(jacP, 0, sizeof(jacP));

  jacS[0][0] = 1;
  jacS[1][1] = 1;
  jacS[2][2] = 1;

  jacT[0][3] = 1;
  jacT[1][4] = 1;
  jacT[2][5] = 1;

  cudaMemcpy(pTildeP, tildeP, numPntBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pS, s, numPntBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pT, t, numPntBytes, cudaMemcpyHostToDevice);

  cudaMemcpy(pJacTildeP, jacTildeP, numJacBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pJacS, jacS, numJacBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pJacT, jacT, numJacBytes, cudaMemcpyHostToDevice);
  */
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  ProjectionOntoLineAndItsJacobian3x6(pTildeP, pS, pT, pJacTildeP, pJacS, pJacT, pP, pJacP, numPoints);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "Time for the kernel: " << time << " ms" << std::endl;

  cudaMemcpy(p, pP + numDims * (numPoints - 1), sizeof(p), cudaMemcpyDeviceToHost);
  cudaMemcpy(jacP, pJacP + numDims * numParams * (numPoints - 1), sizeof(jacP), cudaMemcpyDeviceToHost);

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

  cudaFree(pJacP);
  cudaFree(pJacT);
  cudaFree(pJacS);
  cudaFree(pJacTildeP);
  cudaFree(pP);
  cudaFree(pT);
  cudaFree(pS);
  cudaFree(pTildeP);
}

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