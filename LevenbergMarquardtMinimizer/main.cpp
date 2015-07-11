#include "Minimize.hpp"
#include <H5Cpp.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>

DEFINE_string(source, "", "source filename");
DEFINE_string(result, "", "result filename");
DEFINE_int32(itnlim, 1000, "an upper limit on the number of iterations");

using namespace std;

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

template<typename ValueType>
PredType GetDataType();

template<>
PredType GetDataType<float>() { return PredType::NATIVE_FLOAT; }

template<>
PredType GetDataType<double>() { return PredType::NATIVE_DOUBLE; }

template<>
PredType GetDataType<int>() { return PredType::NATIVE_INT; }

template<typename ValueType>
std::vector<ValueType> Read(const H5File& sourceFile, const string& sourceDataSetName)
{
  auto sourceDataSet = sourceFile.openDataSet(sourceDataSetName);
  vector<ValueType> resultDataSet(sourceDataSet.getSpace().getSimpleExtentNpoints());
  sourceDataSet.read(&resultDataSet[0], GetDataType<ValueType>());
  return resultDataSet;
}

template<typename ValueType>
void Write(H5File& resultFile, const string& resultDataSetName, const std::vector<ValueType>& sourceDataSet)
{
  const int rank = 1;
  const hsize_t dims = sourceDataSet.size();

  auto resultDataSet = resultFile.createDataSet(resultDataSetName, GetDataType<ValueType>(), DataSpace(rank, &dims));
  resultDataSet.write(&sourceDataSet[0], GetDataType<ValueType>());
}

int main(int argc, char *argv[])
{
  const int numDimensions{ 3 };
  typedef float ValueType;
  typedef int IndexType;

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const string sourceFileName{ FLAGS_source };
  const string resultFileName{ FLAGS_result };
  const int itnlim{ FLAGS_itnlim };

  const string measurementsDataSetName{ "measurements" };
  const string tangentLinesPoints1DataSetName{ "tangentLinesPoints1" };
  const string tangentLinesPoints2DataSetName{ "tangentLinesPoints2" };
  const string radiusesDataSetName{ "radiuses" };
  const string positionsDataSetName{ "positions" };
  const string indices1DataSetName{ "indices1" };
  const string indices2DataSetName{ "indices2" };

  const string distancesDataSetName{ "distances" };
  const string curvaturesDataSetName{ "curvatures" };
  
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  LOG(INFO) << "Device " << prop.name << "(" << device << ")";
  LOG(INFO) << "Maximum size of each dimension of a block (" << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << ") ";
  LOG(INFO) << "Maximum number of threads per block " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum resident threads per multiprocessor " << prop.maxThreadsPerMultiProcessor;

  H5File sourceFile(sourceFileName, H5F_ACC_RDONLY);

  auto measurements = Read<ValueType>(sourceFile, measurementsDataSetName);
  auto tangentLinesPoints1 = Read<ValueType>(sourceFile, tangentLinesPoints1DataSetName);
  auto tangentLinesPoints2 = Read<ValueType>(sourceFile, tangentLinesPoints2DataSetName);
  auto radiuses = Read<ValueType>(sourceFile, radiusesDataSetName);
  auto indices1 = Read<IndexType>(sourceFile, indices1DataSetName);
  auto indices2 = Read<IndexType>(sourceFile, indices2DataSetName);

  vector<ValueType> positions(measurements.size());
  
  vector<ValueType> distances(measurements.size() / numDimensions);
  vector<ValueType> curvatures(indices1.size());
  Minimize(&measurements[0], &tangentLinesPoints1[0], &tangentLinesPoints2[0], &radiuses[0], measurements.size() / numDimensions, &indices1[0], &indices2[0], indices1.size(), &positions[0], itnlim, &distances[0], &curvatures[0]);

  H5File resultFile(resultFileName, H5F_ACC_TRUNC);

  Write(resultFile, measurementsDataSetName, measurements);
  Write(resultFile, tangentLinesPoints1DataSetName, tangentLinesPoints1);
  Write(resultFile, tangentLinesPoints2DataSetName, tangentLinesPoints2);
  Write(resultFile, radiusesDataSetName, radiuses);
  Write(resultFile, positionsDataSetName, positions);
  Write(resultFile, indices1DataSetName, indices1);
  Write(resultFile, indices2DataSetName, indices2);
  Write(resultFile, distancesDataSetName, distances);
  Write(resultFile, curvaturesDataSetName, curvatures);
}