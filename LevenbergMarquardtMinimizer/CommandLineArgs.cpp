#include "CommandLineArgs.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(sourceFileName, "", "source filename");
DEFINE_string(resultFileName, "", "result filename");
DEFINE_string(measurementsDataSetName, "measurements", "measurements dataset name");
DEFINE_string(tangentLinesPoints1, "tangentLinesPoints1", "tangent lines points 1 dataset name");
DEFINE_string(tangentLinesPoints2, "tangentLinesPoints2", "tangent lines points 2 dataset name");
DEFINE_string(radiusesDataSetName, "radiuses", "radiuses dataset name");
DEFINE_string(positionsDataSetName, "positions", "positions dataset name");
DEFINE_string(sourcesDataSetName, "sources", "sources dataset name");
DEFINE_string(targetsDataSetName, "targets", "targets dataset name");

DEFINE_int32(nearestNeighbors, 7, "number of nearest neighbors to search");
DEFINE_int32(maxIterations, 1000, "an upper limit on the number of iterations");

const std::string& CommandLineArgs::SourceFileName() const
{
  return FLAGS_sourceFileName;
}

const std::string& CommandLineArgs::ResultFileName() const
{
  return FLAGS_resultFileName;
}

const std::string& CommandLineArgs::MeasurementsDataSetName() const
{
  return FLAGS_measurementsDataSetName;
}

const std::string& CommandLineArgs::TangentsLinesPoints1DataSetName() const
{
  return FLAGS_tangentLinesPoints1;
}

const std::string& CommandLineArgs::TangentsLinesPoints2DataSetName() const
{
  return FLAGS_tangentLinesPoints2;
}

const std::string& CommandLineArgs::RadiusesDataSetName() const
{
  return FLAGS_radiusesDataSetName;
}

const std::string& CommandLineArgs::PositionsDataSetName() const
{
  return FLAGS_positionsDataSetName;
}

const std::string& CommandLineArgs::SourcesDataSetName() const
{
  return FLAGS_sourcesDataSetName;
}

const std::string& CommandLineArgs::TargetsDataSetName() const
{
  return FLAGS_targetsDataSetName;
}

int CommandLineArgs::NearestNeighbors() const
{
  return FLAGS_nearestNeighbors;
}

int CommandLineArgs::MaxIterations() const
{
  return FLAGS_maxIterations;
}

void CommandLineArgs::BriefReport() const
{
  LOG(INFO) << "Source filename: " << SourceFileName();
  LOG(INFO) << "Result filename: " << ResultFileName();
  LOG(INFO) << "Number of nearest neighbors: " << NearestNeighbors();
  LOG(INFO) << "An upper limit on the number of iterations: " << MaxIterations();
}

const CommandLineArgs& CommandLineArgs::Instance()
{
  static CommandLineArgs instance;
  return instance;
}