#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkSignedDanielssonDistanceMapImageFilter.h>
#include <iostream>
#include <string>

void DoDistanceMapImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName)
{
  typedef float ValueType;
  constexpr unsigned int NumDimensions = 3;

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";

  typedef itk::Image<unsigned char, NumDimensions> BinaryImageType;
  typedef itk::Image<ValueType, NumDimensions> ImageType;

  typedef itk::ImageFileReader<BinaryImageType> FileReaderType;
  typedef itk::ImageFileWriter<ImageType> FileWriterType;

  typedef itk::SignedDanielssonDistanceMapImageFilter<BinaryImageType, ImageType> DistanceMapImageFilterType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);

  BinaryImageType::Pointer inputImage =
    imageReader->GetOutput();

  DistanceMapImageFilterType::Pointer distanceMapImageFilter =
    DistanceMapImageFilterType::New();

  distanceMapImageFilter->SetInput(inputImage);
  distanceMapImageFilter->UseImageSpacingOn();
  distanceMapImageFilter->Update();

  ImageType::Pointer outputImage =
    distanceMapImageFilter->GetOutput();

  FileWriterType::Pointer imageWriter =
    FileWriterType::New();

  imageWriter->SetFileName(outputFileName);
  imageWriter->SetInput(outputImage);
  imageWriter->UseCompressionOff();
  imageWriter->Write();
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
    DoDistanceMapImageFilter(inputFileName, outputFileName);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
