#include "PairwiseCostFunction.h"
#include "UnaryCostFunction.h"
#include <iostream>
#include <H5Cpp.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>

#ifndef H5_NO_NAMESPACE

using H5::H5File;
using H5::DataSet;
using H5::DataSpace;
using H5::PredType;

#endif

thrust::host_vector<float> createVector(H5File sourceFile, const H5std_string& targetName)
{
  DataSet targetDataSet = sourceFile.openDataSet(targetName);
  DataSpace targetSpace = targetDataSet.getSpace();
  hssize_t numPoints = targetSpace.getSimpleExtentNpoints();

  thrust::host_vector<float> targetVector(numPoints);
  targetDataSet.read(&targetVector[0], PredType::NATIVE_FLOAT);
  return targetVector;
}

int main(int argc, char *argv[])
{
  H5File sourceFile("D:\\WesternU\\test.h5", H5F_ACC_RDONLY);

  thrust::host_vector<float> px{ createVector(sourceFile, "~p.x") };
  thrust::host_vector<float> py{ createVector(sourceFile, "~p.y") };
  thrust::host_vector<float> pz{ createVector(sourceFile, "~p.z") };

  thrust::host_vector<float> sx{ createVector(sourceFile, "s.x") };
  thrust::host_vector<float> sy{ createVector(sourceFile, "s.y") };
  thrust::host_vector<float> sz{ createVector(sourceFile, "s.z") };

  thrust::host_vector<float> tx{ createVector(sourceFile, "t.x") };
  thrust::host_vector<float> ty{ createVector(sourceFile, "t.y") };
  thrust::host_vector<float> tz{ createVector(sourceFile, "t.z") };

  thrust::host_vector<float> sigma{ createVector(sourceFile, "sigma") };

  auto numPoints = px.size();
  thrust::host_vector<float> unaryCostFunctions(numPoints);

  sourceFile.close();

  //thrust::transform(
  //  thrust::make_zip_iterator(thrust::make_tuple(px.begin(), py.begin(), pz.begin())),
  //  thrust::make_zip_iterator(thrust::make_tuple(px.end(), py.end(), pz.end())),
  //  thrust::make_zip_iterator(thrust::make_tuple(sx.begin(), sy.begin(), sz.begin(), tx.begin(), ty.begin(), tz.begin(), sigma.begin())),
  //  unaryCostFunctions.begin(),
  //  UnaryCostFunction()
  //  );

  //for (int i = 0; i < unaryCostFunctions.size(); ++i)
  //{
  //  std::cout << unaryCostFunctions[i] << std::endl;
  //}

  float unaryCost = thrust::reduce(unaryCostFunctions.begin(), unaryCostFunctions.end());
  std::cout << unaryCost << std::endl;

  auto f = UnaryCostFunction();
  auto grad = UnaryCostFunction::GradientWithRespectToParams();

}