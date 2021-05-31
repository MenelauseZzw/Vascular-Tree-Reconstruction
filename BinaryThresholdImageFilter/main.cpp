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

void DoBinaryThresholdImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName,
  double lowerThreshold,
  double upperThreshold,
  int extractComponentIndex)
{
  typedef double ValueType;
  constexpr unsigned int NumDimensions = 3;
  constexpr unsigned char InsideValue = 255;
  constexpr unsigned char OutsideValue = 0;

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "lower threshold = " << lowerThreshold;
  BOOST_LOG_TRIVIAL(info) << "upper threshold = " << upperThreshold;
  BOOST_LOG_TRIVIAL(info) << "extract component index = " << extractComponentIndex;

  typedef itk::VectorImage<ValueType, NumDimensions> VectorImageType;
  typedef itk::VectorImageToImageAdaptor<ValueType, NumDimensions> VectorImageToImageAdaptorType;
  typedef itk::Image<unsigned char, NumDimensions> BinaryImageType;

  typedef itk::ImageFileReader<VectorImageType> FileReaderType;
  typedef itk::ImageFileWriter<BinaryImageType> FileWriterType;

  typedef itk::BinaryThresholdImageFilter<VectorImageToImageAdaptorType, BinaryImageType> BinaryThresholdImageFilterType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);

  VectorImageType::Pointer inputImage =
    imageReader->GetOutput();

  VectorImageToImageAdaptorType::Pointer imageAdapter =
    VectorImageToImageAdaptorType::New();

  imageAdapter->SetImage(inputImage);
  imageAdapter->SetExtractComponentIndex(0);
  imageAdapter->Update();

  BOOST_LOG_TRIVIAL(info) << "number of components per pixel = " << inputImage->GetNumberOfComponentsPerPixel();

  BinaryThresholdImageFilterType::Pointer binaryThresholdImageFilter =
    BinaryThresholdImageFilterType::New();

  binaryThresholdImageFilter->SetInput(imageAdapter);
  binaryThresholdImageFilter->SetInsideValue(InsideValue);
  binaryThresholdImageFilter->SetOutsideValue(OutsideValue);
  binaryThresholdImageFilter->SetLowerThreshold(lowerThreshold);
  binaryThresholdImageFilter->SetUpperThreshold(upperThreshold);
  binaryThresholdImageFilter->Update();

  BinaryImageType::Pointer outputImage =
    binaryThresholdImageFilter->GetOutput();

  typedef itk::MetaDataDictionary MetaDataDictionaryType;

  MetaDataDictionaryType outMetaData;

  EncapsulateMetaData(outMetaData, "(LowerThreshold)", lowerThreshold);
  EncapsulateMetaData(outMetaData, "(UpperThreshold)", upperThreshold);

  outputImage->SetMetaDataDictionary(outMetaData);

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

  double lowerThreshold = 0.0;
  double upperThreshold = 1.0;
  int extractComponentIndex = 0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("lowerThreshold", po::value(&lowerThreshold), "the lower threshold value")
    ("upperThreshold", po::value(&upperThreshold), "the upper threshold value")
    ("extractComponentIndex", po::value(&extractComponentIndex), "the component to be extracted")
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
    DoBinaryThresholdImageFilter(inputFileName, outputFileName, lowerThreshold, upperThreshold, extractComponentIndex);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
