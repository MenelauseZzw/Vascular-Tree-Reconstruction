#include "TestLevenbergMarquardtMinimizer1.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <H5Cpp.h>
#include <iostream>
#include <vector>

DEFINE_string(i, "", "");
DEFINE_string(o, "", "");

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

template<typename T>
PredType TypeOf();

template<>
PredType TypeOf<float>() { return PredType::NATIVE_FLOAT; }

template<>
PredType TypeOf<int>() { return PredType::NATIVE_INT; }

template<typename T>
std::vector<T> readVector(H5File sourceFile, const H5std_string& targetName)
{
  DataSet targetDataSet = sourceFile.openDataSet(targetName);
  DataSpace targetSpace = targetDataSet.getSpace();

  std::vector<T> targetVector(targetSpace.getSimpleExtentNpoints());
  targetDataSet.read(&targetVector[0], TypeOf<T>());
  return targetVector;
}

int main(int argc, char *argv[])
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  H5File sourceFile(FLAGS_i, H5F_ACC_RDONLY);

  std::vector<float> tildeP = readVector<float>(sourceFile, "tildeP");
  std::vector<float> s = readVector<float>(sourceFile, "s");
  std::vector<float> t = readVector<float>(sourceFile, "t");
  std::vector<int> indPi = readVector<int>(sourceFile, "indPi");
  std::vector<int> indPj = readVector<int>(sourceFile, "indPj");
  std::vector<float> sigma = readVector<float>(sourceFile, "sigma");
  std::vector<float> p(tildeP.size());

  testLevenbergMarquardtMinimizer1(&tildeP[0], &s[0], &t[0], &sigma[0], tildeP.size() / 3, &indPi[0], &indPj[0], indPi.size(), &p[0]);

  H5File resultFile(FLAGS_o, H5F_ACC_TRUNC);

  const int rank = 1;
  {
    const hsize_t dims[rank] = { tildeP.size() };
    DataSpace space(rank, dims);

    resultFile.createDataSet("tildeP", PredType::NATIVE_FLOAT, space).write(&tildeP[0], PredType::NATIVE_FLOAT);
    resultFile.createDataSet("s", PredType::NATIVE_FLOAT, space).write(&s[0], PredType::NATIVE_FLOAT);
    resultFile.createDataSet("t", PredType::NATIVE_FLOAT, space).write(&t[0], PredType::NATIVE_FLOAT);
    resultFile.createDataSet("p", PredType::NATIVE_FLOAT, space).write(&p[0], PredType::NATIVE_FLOAT);
  }
  {
    const hsize_t dims[rank] = { indPi.size() };
    DataSpace space(rank, dims);
    
    resultFile.createDataSet("indPi", PredType::NATIVE_INT, space).write(&indPi[0], PredType::NATIVE_INT);
    resultFile.createDataSet("indPj", PredType::NATIVE_INT, space).write(&indPj[0], PredType::NATIVE_INT);
  }
  {
    const hsize_t dims[rank] = { sigma.size() };
    DataSpace space(rank, dims);

    resultFile.createDataSet("sigma", PredType::NATIVE_FLOAT, space).write(&sigma[0], PredType::NATIVE_FLOAT);
  }
  resultFile.close();
}