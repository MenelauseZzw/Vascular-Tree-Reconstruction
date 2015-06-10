#include "GraphDataWriter.hpp"
#include "GraphData.hpp"
#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

static PredType GetPredType(const thrust::host_vector<float>&)
{
  return PredType::NATIVE_FLOAT;
}

static PredType GetPredType(const thrust::host_vector<int>&)
{
  return PredType::NATIVE_INT;
}

template<typename ValueType>
static void WriteDataSet(H5File& targetFile, const std::string& targetDataSetName, const thrust::host_vector<ValueType>& sourceVector)
{
  const int rank = 1;
  const hsize_t dims = sourceVector.size();
  
  DataSpace targetSpace(rank, &dims);
  DataSet targetDataSet = targetFile.createDataSet(targetDataSetName, GetPredType(sourceVector), targetSpace);
  
  targetDataSet.write(&sourceVector[0], GetPredType(sourceVector));
}

template<typename ValueType, typename IndexType>
void GraphDataWriter<ValueType, IndexType>::Write(const Options& writerOptions, const GraphData<ValueType, IndexType>& graphData) const
{
  H5File targetFile(writerOptions.targetFileName, H5F_ACC_TRUNC);

  WriteDataSet(targetFile, writerOptions.measurementsDataSetName, graphData.measurements);
  WriteDataSet(targetFile, writerOptions.tangentsLines1PointsDataSetName, graphData.tangentsLinesPoints1);
  WriteDataSet(targetFile, writerOptions.tangentsLines2PointsDataSetName, graphData.tangentsLinesPoints2);
  WriteDataSet(targetFile, writerOptions.radiusesDataSetName, graphData.radiuses);
  WriteDataSet(targetFile, writerOptions.positionsDataSetName, graphData.positions);
  WriteDataSet(targetFile, writerOptions.sourcesDataSetName, graphData.sources);
  WriteDataSet(targetFile, writerOptions.targetsDataSetName, graphData.targets);
}

template struct GraphDataWriter<float, int>;