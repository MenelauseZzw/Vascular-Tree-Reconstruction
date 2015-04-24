#include "CommandLineArgs.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_string(i, "", "source filename");
DEFINE_string(o, "", "result filename");
DEFINE_int32(nn, 7, "number of nearest neighbors to search");

const std::string& CommandLineArgs::SourceFileName()
{
  return FLAGS_i;
}

const std::string& CommandLineArgs::ResultFileName()
{
  return FLAGS_o;
}

int CommandLineArgs::NearestNeighbors()
{
  return FLAGS_nn;
}

void CommandLineArgs::Notify()
{
  LOG(INFO) << "Source filename: " << SourceFileName();
  LOG(INFO) << "Result filename: " << ResultFileName();
  LOG(INFO) << "Number of nearest neighbors: " << NearestNeighbors();
}