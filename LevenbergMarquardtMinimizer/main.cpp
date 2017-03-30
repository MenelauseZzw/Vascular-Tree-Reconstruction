// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "FileReader.hpp"
#include "FileWriter.hpp"
#include "Minimize.hpp"
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <vector>

void DoLevenbergMarquardtMinimizer(const std::string& inputFileName, const std::string& outputFileName, double lambda, int maxNumberOfIterations, int gpuDevice)
{
  const unsigned int NumDimensions = 3;

  typedef double ValueType;
  typedef int IndexType;

  const std::string measurementsDataSetName = "measurements";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";
  const std::string positionsDataSetName = "positions";

  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";

  FileReader inputFileReader(inputFileName);

  std::vector<ValueType> measurements;
  std::vector<ValueType> tangentLinesPoints1;
  std::vector<ValueType> tangentLinesPoints2;
  std::vector<ValueType> radiuses;

  std::vector<IndexType> indices1;
  std::vector<IndexType> indices2;

  inputFileReader.Read(measurementsDataSetName, measurements);
  inputFileReader.Read(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  inputFileReader.Read(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  inputFileReader.Read(radiusesDataSetName, radiuses);

  inputFileReader.Read(indices1DataSetName, indices1);
  inputFileReader.Read(indices2DataSetName, indices2);

  std::vector<ValueType> lambdas(indices1.size(), lambda);
  std::vector<ValueType> positions(measurements.size());

  if (gpuDevice != -1)
  {
    if (cudaErrorInvalidDevice == cudaSetDevice(gpuDevice))
      cudaGetDevice(&gpuDevice);

    cudaDeviceProp gpuDeviceProp;
    cudaGetDeviceProperties(&gpuDeviceProp, gpuDevice);

    BOOST_LOG_TRIVIAL(info) << "Device " << gpuDeviceProp.name << "(" << gpuDevice << ")";
    BOOST_LOG_TRIVIAL(info) << "Maximum size of each dimension of a block (" << gpuDeviceProp.maxThreadsDim[0] << " " << gpuDeviceProp.maxThreadsDim[1] << " " << gpuDeviceProp.maxThreadsDim[2] << ") ";
    BOOST_LOG_TRIVIAL(info) << "Maximum number of threads per block " << gpuDeviceProp.maxThreadsPerBlock;
    BOOST_LOG_TRIVIAL(info) << "Maximum resident threads per multiprocessor " << gpuDeviceProp.maxThreadsPerMultiProcessor;

    //DoGpuLevenbergMarquardtMinimizer(measurements, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, positions, lambdas, maxNumberOfIterations);
  }
  else
  {
    //DoCpuLevenbergMarquardtMinimizer(measurements, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, positions, lambdas, maxNumberOfIterations);
  }

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(measurementsDataSetName, measurements);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
  outputFileWriter.Write(positionsDataSetName, positions);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

 
  int maxNumberOfIterations = 1000;
  double lambda;
  std::string inputFileName;
  std::string outputFileName; 
  int gpuDevice = -1;

  po::options_description desc;
  desc.add_options()
    ("help", "print usage message")
    ("lambda", po::value(&lambda)->required(), "the value of regularization parameter")
    ("gpuDevice", po::value(&gpuDevice), "gpu device should be used; otherwise, run on cpu")
    ("maxNumberOfIterations", po::value(&maxNumberOfIterations), "the upper limit on the number of iterations")
    ("inputFileName", po::value(&inputFileName)->required(), "the name of the input file")
    ("outputFileName", po::value(&outputFileName)->required(), "the name of the output file");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    desc.print(std::cout);
    return EXIT_SUCCESS;
  }

  try
  {
    DoLevenbergMarquardtMinimizer(inputFileName, outputFileName, lambda, maxNumberOfIterations, gpuDevice);
    return EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
