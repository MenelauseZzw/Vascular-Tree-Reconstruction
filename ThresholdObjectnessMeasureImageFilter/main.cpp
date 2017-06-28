#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkThresholdImageFilter.h>
#include <itkVectorImage.h>
#include <itkVectorImageToImageAdaptor.h>
#include <iostream>
#include <string>

void DoThresholdObjectnessMeasureImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName,
  double lowerThreshold,
  double upperThreshold,
  double outsideValue)
{
  typedef double ValueType;

  constexpr unsigned int NumDimensions = 3;

  constexpr unsigned int ObjectnessMeasureValueComponentIndex = 0;
  constexpr unsigned int SigmaValueC= 1 + NumDimensions;

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "lower threshold = " << lowerThreshold;
  BOOST_LOG_TRIVIAL(info) << "upper threshold = " << upperThreshold;
  BOOST_LOG_TRIVIAL(info) << "outside value = " << outsideValue;

  typedef itk::Index<NumDimensions> IndexType;

  typedef itk::VectorImage<ValueType, NumDimensions> VectorImageType;
  typedef itk::Image<ValueType, NumDimensions> OutputImageType;
  typedef itk::ImageRegionIterator<VectorImageType> ImageIteratorType;

  typedef itk::ImageFileReader<VectorImageType> FileReaderType;
  typedef itk::ImageFileWriter<OutputImageType> FileWriterType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  VectorImageType::Pointer inputImage =
    imageReader->GetOutput();

  ImageIteratorType it(inputImage, inputImage->GetLargestPossibleRegion());

  OutputImageType::Pointer outputImage =
    OutputImageType::New();

  outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
  outputImage->Allocate();

  outputImage->SetOrigin(inputImage->GetOrigin());
  outputImage->SetSpacing(inputImage->GetSpacing());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const IndexType index = it.GetIndex();
    const ValueType objectnessMeasureValue = inputImage->GetPixel(index).GetElement(ObjectnessMeasureValueComponentIndex);

    if (objectnessMeasureValue >= lowerThreshold && objectnessMeasureValue <= upperThreshold)
    {
      outputImage->SetPixel(index, objectnessMeasureValue);
    }
    else
    {
      outputImage->SetPixel(index, outsideValue);
    }
  }

  typedef itk::MetaDataDictionary MetaDataDictionaryType;

  MetaDataDictionaryType outMetaData;

  EncapsulateMetaData(outMetaData, "(LowerThreshold)", lowerThreshold);
  EncapsulateMetaData(outMetaData, "(UpperThreshold)", upperThreshold);
  EncapsulateMetaData(outMetaData, "(OutsideValue)", outsideValue);

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
  double outsideValue = 0.0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("lowerThreshold", po::value(&lowerThreshold), "the lower threshold value")
    ("upperThreshold", po::value(&upperThreshold), "the upper threshold value")
    ("outsideValue", po::value(&outsideValue), "the \"outside\" pixel value")
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
    DoThresholdObjectnessMeasureImageFilter(inputFileName, outputFileName, lowerThreshold, upperThreshold, outsideValue);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
