#include <boost/program_options.hpp>
#include <itkAdditiveGaussianNoiseImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <iostream>
#include <string>

void DoAdditiveGaussianNoiseImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName, double mean, double standardDeviation)
{
  typedef short ValueType;
  constexpr unsigned int NumDimensions = 3;

  typedef itk::Image<ValueType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef typename ImageType::SizeType SizeType;

  typedef itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType> AdditiveGaussianNoiseImageFilterType;
  typedef itk::ImageFileReader<ImageType> FileReaderType;
  typedef itk::ImageFileWriter<ImageType> FileWriterType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);

  ImageType::Pointer inputImage =
    imageReader->GetOutput();

  AdditiveGaussianNoiseImageFilterType::Pointer additiveGaussianNoiseImageFilter =
    AdditiveGaussianNoiseImageFilterType::New();

  additiveGaussianNoiseImageFilter->SetInput(inputImage);
  additiveGaussianNoiseImageFilter->SetMean(mean);
  additiveGaussianNoiseImageFilter->SetStandardDeviation(standardDeviation);
  additiveGaussianNoiseImageFilter->Update();

  ImageType::Pointer outputImage = 
    additiveGaussianNoiseImageFilter->GetOutput();

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

  double mean = 0;
  double standardDeviation = 1;

  desc.add_options()
    ("help", "print usage message")
    ("mean", po::value(&mean), "the mean of the Gaussian distribution")
    ("standardDeviation", po::value(&standardDeviation), "the standard deviation of the Gaussian distribution")
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
    DoAdditiveGaussianNoiseImageFilter(inputFileName, outputFileName, mean, standardDeviation);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
