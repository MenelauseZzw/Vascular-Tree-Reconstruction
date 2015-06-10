#include "GraphDataReader.hpp"
#include "GraphData.hpp"
#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

static PredType GetPredType(const thrust::host_vector<float>&)
{
  return PredType::NATIVE_FLOAT;
}

template<typename ValueType>
static void ReadDataSet(H5File& sourceFile, const std::string& sourceDataSetName, thrust::host_vector<ValueType>& targetVector)
{
  DataSet sourceDataSet = sourceFile.openDataSet(sourceDataSetName);
  DataSpace sourceSpace = sourceDataSet.getSpace();

  thrust::host_vector<ValueType> tmpVector(sourceSpace.getSimpleExtentNpoints());

  sourceDataSet.read(&tmpVector[0], GetPredType(tmpVector));
  tmpVector.swap(targetVector);
}

template<typename ValueType, typename IndexType>
void GraphDataReader<ValueType, IndexType>::Read(const Options& readerOptions, GraphData<ValueType, IndexType>& graphData) const
{
  H5File sourceFile(readerOptions.sourceFileName, H5F_ACC_RDONLY);

  GraphData<ValueType, IndexType> tmpGraphData;

  ReadDataSet(sourceFile, readerOptions.measurementsDataSetName, tmpGraphData.measurements);
  ReadDataSet(sourceFile, readerOptions.tangentsLinesPoints1DataSetName, tmpGraphData.tangentsLinesPoints1);
  ReadDataSet(sourceFile, readerOptions.tangentsLinesPoints2DataSetName, tmpGraphData.tangentsLinesPoints2);
  ReadDataSet(sourceFile, readerOptions.radiusesDataSetName, tmpGraphData.radiuses);

  tmpGraphData.Swap(graphData);
}

template struct GraphDataReader<float, int>;