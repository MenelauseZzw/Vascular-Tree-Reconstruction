#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkIndex.h>
#include <itkVectorImage.h>
#include <fstream>
#include <string>

void DoGenerateSeedListFile(
  const std::string& inputFileName,
  const std::string& outputFileName,
  double lowerThreshold,
  double upperThreshold)
{
  typedef double ValueType;

  constexpr unsigned int NumDimensions = 3;

  constexpr unsigned int ObjectnessMeasureValueComponentIndex = 0;
  constexpr unsigned int ObjectnessMeasureTangentsComponentIndex = 1;
  constexpr unsigned int SigmaValueComponentIndex = 1 + NumDimensions;

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "lower threshold = " << lowerThreshold;
  BOOST_LOG_TRIVIAL(info) << "upper threshold = " << upperThreshold;

  typedef itk::Index<NumDimensions> IndexType;

  typedef itk::VectorImage<ValueType, NumDimensions> ImageType;
  typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;

  typedef itk::ImageFileReader<ImageType> FileReaderType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ImageType::Pointer inputImage =
    imageReader->GetOutput();
  
  std::ofstream outputFile(outputFileName);

  ImageIteratorType it(inputImage, inputImage->GetLargestPossibleRegion());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const IndexType indexOfCenter = it.GetIndex();
    const ValueType objectnessMeasureValueAtCenter = inputImage->GetPixel(indexOfCenter).GetElement(ObjectnessMeasureValueComponentIndex);
    const ValueType sigmaValueAtCenter = inputImage->GetPixel(indexOfCenter).GetElement(SigmaValueComponentIndex);
    
    if (objectnessMeasureValueAtCenter >= lowerThreshold && objectnessMeasureValueAtCenter <= upperThreshold)
    {
      outputFile << indexOfCenter[0] << " " << indexOfCenter[1] << " " << indexOfCenter[2] << " " << sigmaValueAtCenter << std::endl;
    }
  }
}

int main(int argc, char* argv[])
{
  namespace po = boost::program_options;

  std::string inputFileName;
  std::string outputFileName;

  double lowerThreshold = 0.0;
  double upperThreshold = 1.0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("lowerThreshold", po::value(&lowerThreshold), "the lower threshold value")
    ("upperThreshold", po::value(&upperThreshold), "the upper threshold value")
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
    DoGenerateSeedListFile(inputFileName, outputFileName, lowerThreshold, upperThreshold);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
