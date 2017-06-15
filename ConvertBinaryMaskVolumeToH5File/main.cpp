#include "FileWriter.hpp"
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkBinaryThresholdImageFilter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorImage.h>
#include <itkVectorImageToImageAdaptor.h>
#include <iostream>
#include <string>

void DoConvertBinaryMaskVolumeToH5File(
  const std::string& inputFileName,
  const std::string& outputFileName)
{
  typedef unsigned char InValueType;
  typedef double OutValueType;
  constexpr unsigned int NumDimensions = 3;

  const std::string positionsDataSetName = "positions";

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";

  typedef itk::Image<InValueType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef itk::Point<OutValueType, NumDimensions> PointType;
  typedef itk::ImageFileReader<ImageType> FileReaderType;
  typedef itk::ImageRegionConstIterator<ImageType> ImageIteratorType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ImageType::ConstPointer inputImage =
    imageReader->GetOutput();

  ImageIteratorType it(inputImage, inputImage->GetLargestPossibleRegion());

  std::vector<OutValueType> positions;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const IndexType indexOfPixel = it.GetIndex();

    if (it.Get())
    {
      PointType pointAtPixel;
      inputImage->TransformIndexToPhysicalPoint(indexOfPixel, pointAtPixel);

      positions.insert(positions.end(), pointAtPixel.Begin(), pointAtPixel.End());
    }
  }

  BOOST_LOG_TRIVIAL(info) << "number of points = " << positions.size() / NumDimensions;

  FileWriter outputFileWriter(outputFileName);
  
  outputFileWriter.Write(positionsDataSetName, positions);
}

int main(int argc, char* argv[])
{
  namespace po = boost::program_options;

  std::string inputFileName;
  std::string outputFileName;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
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
    DoConvertBinaryMaskVolumeToH5File(inputFileName, outputFileName);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
