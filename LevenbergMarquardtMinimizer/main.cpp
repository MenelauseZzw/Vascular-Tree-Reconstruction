#include "TestLsqr.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <H5Cpp.h>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  testLsqr();
}

//
//#ifndef H5_NO_NAMESPACE
//using namespace H5;
//#endif
//
//template<typename T>
//PredType TypeOf();
//
//template<>
//PredType TypeOf<float>() { return PredType::NATIVE_FLOAT; }
//
//template<>
//PredType TypeOf<int>() { return PredType::NATIVE_INT; }
//
//template<typename T>
//std::vector<T> readVector(H5File sourceFile, const H5std_string& targetName)
//{
//  DataSet targetDataSet = sourceFile.openDataSet(targetName);
//  DataSpace targetSpace = targetDataSet.getSpace();
//
//  std::vector<T> targetVector(targetSpace.getSimpleExtentNpoints());
//  targetDataSet.read(&targetVector[0], TypeOf<T>());
//  return targetVector;
//}
//
//int main(int argc, char *argv[])
//{
//  google::InitGoogleLogging(argv[0]);
//  gflags::ParseCommandLineFlags(&argc, &argv, true);
//
//  H5File sourceFile("C:\\WesternU\\test.h5", H5F_ACC_RDONLY);
//
//  std::vector<float> tildeP = readVector<float>(sourceFile, "tildeP");
//  std::vector<float> s = readVector<float>(sourceFile, "s");
//  std::vector<float> t = readVector<float>(sourceFile, "t");
//  std::vector<int> indPi = readVector<int>(sourceFile, "indPi");
//  std::vector<int> indPj = readVector<int>(sourceFile, "indPj");
//  std::vector<float> sigma = readVector<float>(sourceFile, "sigma");
//  std::vector<float> p(tildeP.size());
//
//  testLevenbergMarquardtMinimizer(&tildeP[0], &s[0], &t[0], &sigma[0], tildeP.size() / 3, &indPi[0], &indPj[0], indPi.size(), &p[0]);
//
//  H5File resultFile("C:\\WesternU\\testResult.h5", H5F_ACC_TRUNC);
//
//  const int rank = 1;
//  {
//    const hsize_t dims[rank] = { tildeP.size() };
//    DataSpace space(rank, dims);
//
//    resultFile.createDataSet("tildeP", PredType::NATIVE_FLOAT, space).write(&tildeP[0], PredType::NATIVE_FLOAT);
//    resultFile.createDataSet("s", PredType::NATIVE_FLOAT, space).write(&s[0], PredType::NATIVE_FLOAT);
//    resultFile.createDataSet("t", PredType::NATIVE_FLOAT, space).write(&t[0], PredType::NATIVE_FLOAT);
//    resultFile.createDataSet("p", PredType::NATIVE_FLOAT, space).write(&p[0], PredType::NATIVE_FLOAT);
//  }
//  {
//    const hsize_t dims[rank] = { indPi.size() };
//    DataSpace space(rank, dims);
//    
//    resultFile.createDataSet("indPi", PredType::NATIVE_INT, space).write(&indPi[0], PredType::NATIVE_INT);
//    resultFile.createDataSet("indPj", PredType::NATIVE_INT, space).write(&indPj[0], PredType::NATIVE_INT);
//  }
//  resultFile.close();
//}

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