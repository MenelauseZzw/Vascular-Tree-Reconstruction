#include <boost/program_options.hpp>
#include <itkCropImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <iostream>
#include <string>

void DoCropImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName,
  unsigned int lowerBoundaryCropSizeX, unsigned int lowerBoundaryCropSizeY, unsigned int lowerBoundaryCropSizeZ, 
  unsigned int upperBoundaryCropSizeX, unsigned int upperBoundaryCropSizeY, unsigned int upperBoundaryCropSizeZ)
{
  typedef short ValueType;
  constexpr unsigned int NumDimensions = 3;

  typedef itk::Image<ValueType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef typename ImageType::SizeType SizeType;

  typedef itk::CropImageFilter<ImageType, ImageType> CropImageFilterType;
  typedef itk::ImageFileReader<ImageType> FileReaderType;
  typedef itk::ImageFileWriter<ImageType> FileWriterType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);

  ImageType::Pointer inputImage =
    imageReader->GetOutput();

  CropImageFilterType::Pointer cropImageFile =
    CropImageFilterType::New();

  cropImageFile->SetInput(imageReader->GetOutput());

  SizeType lowerBoundaryCropSize;
  
  lowerBoundaryCropSize.SetElement(0, lowerBoundaryCropSizeX);
  lowerBoundaryCropSize.SetElement(1, lowerBoundaryCropSizeY);
  lowerBoundaryCropSize.SetElement(2, lowerBoundaryCropSizeZ);

  SizeType upperBoundaryCropSize;

  upperBoundaryCropSize.SetElement(0, upperBoundaryCropSizeX);
  upperBoundaryCropSize.SetElement(1, upperBoundaryCropSizeY);
  upperBoundaryCropSize.SetElement(2, upperBoundaryCropSizeZ);
  
  cropImageFile->SetLowerBoundaryCropSize(lowerBoundaryCropSize);
  cropImageFile->SetUpperBoundaryCropSize(upperBoundaryCropSize);
  cropImageFile->Update();

  ImageType::Pointer outputImage = 
    cropImageFile->GetOutput();

  outputImage->SetMetaDataDictionary(inputImage->GetMetaDataDictionary());

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

  unsigned int lowerBoundaryCropSizeX = 0;
  unsigned int lowerBoundaryCropSizeY = 0;
  unsigned int lowerBoundaryCropSizeZ = 0;

  unsigned int upperBoundaryCropSizeX = 0;
  unsigned int upperBoundaryCropSizeY = 0;
  unsigned int upperBoundaryCropSizeZ = 0;

  desc.add_options()
    ("help", "print usage message")
    ("lowerBoundaryCropSizeX", po::value(&lowerBoundaryCropSizeX), "the cropping sizes for the lower boundaries (X)")
    ("lowerBoundaryCropSizeY", po::value(&lowerBoundaryCropSizeY), "the cropping sizes for the lower boundaries (Y)")
    ("lowerBoundaryCropSizeZ", po::value(&lowerBoundaryCropSizeZ), "the cropping sizes for the lower boundaries (Z)")
    ("upperBoundaryCropSizeX", po::value(&upperBoundaryCropSizeX), "the cropping sizes for the upper boundaries (X)")
    ("upperBoundaryCropSizeY", po::value(&upperBoundaryCropSizeY), "the cropping sizes for the upper boundaries (Y)")
    ("upperBoundaryCropSizeZ", po::value(&upperBoundaryCropSizeZ), "the cropping sizes for the upper boundaries (Z)")
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
    DoCropImageFilter(inputFileName, outputFileName, lowerBoundaryCropSizeX, lowerBoundaryCropSizeY, lowerBoundaryCropSizeZ, upperBoundaryCropSizeX, upperBoundaryCropSizeY, upperBoundaryCropSizeZ);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
