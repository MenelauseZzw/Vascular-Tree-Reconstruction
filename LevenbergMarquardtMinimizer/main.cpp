#include "CommandLineArgs.h"
#include "GraphData.hpp"
#include "GraphDataReader.hpp"
#include "GraphDataWriter.hpp"
#include "TestKnnSearch1.h"
#include "TestLevenbergMarquardtMinimizer1.h"
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <vector>

int main(int argc, char *argv[])
{
  typedef float ValueType;
  typedef int IndexType;

  typedef GraphData<ValueType, IndexType> GraphDataType;
  typedef GraphDataReader<ValueType, IndexType> GraphDataReaderType;
  typedef GraphDataWriter<ValueType, IndexType> GraphDataWriterType;

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const CommandLineArgs& commandLineArgs = CommandLineArgs::Instance();
  commandLineArgs.BriefReport();
  
  GraphDataType graphData;
  GraphDataReaderType::Options readerOptions;

  readerOptions.sourceFileName = commandLineArgs.SourceFileName();
  readerOptions.measurementsDataSetName = commandLineArgs.MeasurementsDataSetName();
  readerOptions.tangentsLinesPoints1DataSetName = commandLineArgs.TangentsLinesPoints1DataSetName();
  readerOptions.tangentsLinesPoints2DataSetName = commandLineArgs.TangentsLinesPoints2DataSetName();
  readerOptions.radiusesDataSetName = commandLineArgs.RadiusesDataSetName();

  GraphDataReaderType reader;
  reader.Read(readerOptions, graphData);

  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  LOG(INFO) << "Device " << prop.name << "(" << device << ")";
  LOG(INFO) << "Maximum size of each dimension of a block (" << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << ") ";
  LOG(INFO) << "Maximum number of threads per block " << prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum resident threads per multiprocessor " << prop.maxThreadsPerMultiProcessor;

  graphData.sources.resize(commandLineArgs.NearestNeighbors() * graphData.radiuses.size() + 1);
  graphData.targets.resize(graphData.sources.size());

  int numPairs;

  testKnnSearch(&graphData.measurements[0], &graphData.radiuses[0], &graphData.tangentsLinesPoints1[0], &graphData.tangentsLinesPoints2[0], &graphData.sources[0], &graphData.targets[0], graphData.radiuses.size(), commandLineArgs.NearestNeighbors(), numPairs);
  LOG(INFO) << "Number of pairwise terms: " << numPairs;

  graphData.sources.resize(numPairs);
  graphData.targets.resize(numPairs);
  
  graphData.positions.resize(graphData.measurements.size());
  testLevenbergMarquardtMinimizer1(&graphData.measurements[0], &graphData.tangentsLinesPoints1[0], &graphData.tangentsLinesPoints2[0], &graphData.radiuses[0], graphData.radiuses.size(), &graphData.sources[0], &graphData.targets[0], graphData.targets.size(), &graphData.positions[0], CommandLineArgs::Instance().MaxIterations());

  GraphDataWriterType::Options writerOptions;

  writerOptions.targetFileName = commandLineArgs.ResultFileName();
  writerOptions.measurementsDataSetName = commandLineArgs.MeasurementsDataSetName();
  writerOptions.tangentsLines1PointsDataSetName = commandLineArgs.TangentsLinesPoints1DataSetName();
  writerOptions.tangentsLines2PointsDataSetName = commandLineArgs.TangentsLinesPoints2DataSetName();
  writerOptions.radiusesDataSetName = commandLineArgs.RadiusesDataSetName();
  writerOptions.positionsDataSetName = commandLineArgs.PositionsDataSetName();
  writerOptions.sourcesDataSetName = commandLineArgs.SourcesDataSetName();
  writerOptions.targetsDataSetName = commandLineArgs.TargetsDataSetName();

  GraphDataWriterType writer;
  writer.Write(writerOptions, graphData);
}