#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkAdditiveGaussianNoiseImageFilter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkSquaredDifferenceImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <iostream>
#include <string>

void DoAdditiveGaussianNoiseImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName, double mean, double standardDeviation)
{
  typedef double ValueType;
  constexpr unsigned int NumDimensions = 3;

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";

  BOOST_LOG_TRIVIAL(info) << "mean = " << mean;
  BOOST_LOG_TRIVIAL(info) << "standardDeviation = " << standardDeviation;

  typedef itk::Image<ValueType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef typename ImageType::SizeType SizeType;

  typedef itk::AdditiveGaussianNoiseImageFilter<ImageType, ImageType> AdditiveGaussianNoiseImageFilterType;
  typedef itk::SquaredDifferenceImageFilter<ImageType, ImageType, ImageType> SquaredDifferenceImageFilterType;
  typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
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

  SquaredDifferenceImageFilterType::Pointer squaredDifferenceImageFilter =
    SquaredDifferenceImageFilterType::New();

  squaredDifferenceImageFilter->SetInput1(inputImage);
  squaredDifferenceImageFilter->SetInput2(outputImage);
  squaredDifferenceImageFilter->Update();

  StatisticsImageFilterType::Pointer statisticsImageFilter =
    StatisticsImageFilterType::New();

  statisticsImageFilter->SetInput(squaredDifferenceImageFilter->GetOutput());
  statisticsImageFilter->Update();

  const ValueType meanSquareError = statisticsImageFilter->GetMean();

  statisticsImageFilter->SetInput(outputImage);
  statisticsImageFilter->Update();

  const ValueType maxValue = statisticsImageFilter->GetMaximum();
  const ValueType peakSignalToNoiseRatioDecibel = 10 * log10(maxValue * maxValue / meanSquareError);

  typedef itk::MetaDataDictionary MetaDataDictionaryType;

  BOOST_LOG_TRIVIAL(info) << "Mean Square Error  = " << meanSquareError;
  BOOST_LOG_TRIVIAL(info) << "Peak Signal To Noise Ratio = " << peakSignalToNoiseRatioDecibel << " dB";

  MetaDataDictionaryType outMetaData;

  EncapsulateMetaData(outMetaData, "(Mean)", mean);
  EncapsulateMetaData(outMetaData, "(StandardDeviation)", standardDeviation);
  EncapsulateMetaData(outMetaData, "(MeanSquareError)", meanSquareError);
  EncapsulateMetaData(outMetaData, "(PeakSignalToNoiseRatioDecibel)", peakSignalToNoiseRatioDecibel);

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
