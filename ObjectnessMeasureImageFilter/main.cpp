#include <boost/program_options.hpp>
#include <iostream>
#include <itkComposeImageFilter.h>
#include <itkHessianToObjectnessMeasureImageFilter.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkIndex.h>
#include <itkMacro.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkMultiScaleHessianBasedMeasureImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkSymmetricSecondRankTensor.h>
#include <itkVectorImage.h>
#include <string>

void DoObjectnessMeasureImageFilter(
  const std::string& inputFileName,
  const std::string& outputFileName,
  double thresholdBelow,
  double sigmaMaximum,
  double sigmaMinimum,
  unsigned int numberOfSigmaSteps,
  double alpha,
  double beta,
  double gamma,
  unsigned int objectDimension,
  bool scaleObjectnessMeasure,
  double outputMaximum,
  double outputMinimum)
{
  typedef float ValueType;
  constexpr unsigned int NumDimensions = 3;

  typedef itk::Image<ValueType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;

  typedef itk::SymmetricSecondRankTensor<ValueType, NumDimensions> HessianValueType;
  typedef itk::Image<HessianValueType, NumDimensions> HessianImageType;

  typedef itk::ImageFileReader<ImageType> FileReaderType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ImageType::ConstPointer inputImage =
    imageReader->GetOutput();

  typedef itk::MultiScaleHessianBasedMeasureImageFilter<ImageType, HessianImageType> MultiScaleEnhancementFilterType;
  typedef itk::HessianToObjectnessMeasureImageFilter<HessianImageType, ImageType> ObjectnessFilterType;

  MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter =
    MultiScaleEnhancementFilterType::New();

  multiScaleEnhancementFilter->GenerateHessianOutputOn();
  multiScaleEnhancementFilter->GenerateScalesOutputOn();
  multiScaleEnhancementFilter->SetInput(imageReader->GetOutput());
  multiScaleEnhancementFilter->SetNumberOfSigmaSteps(numberOfSigmaSteps);
  multiScaleEnhancementFilter->SetSigmaMaximum(sigmaMaximum);
  multiScaleEnhancementFilter->SetSigmaMinimum(sigmaMinimum);
  multiScaleEnhancementFilter->SetSigmaStepMethodToLogarithmic();
  multiScaleEnhancementFilter->NonNegativeHessianBasedMeasureOn();

  ObjectnessFilterType::Pointer objectnessFilter =
    ObjectnessFilterType::New();

  multiScaleEnhancementFilter->SetHessianToMeasureFilter(objectnessFilter);

  objectnessFilter->BrightObjectOn();
  objectnessFilter->SetAlpha(alpha);
  objectnessFilter->SetBeta(beta);
  objectnessFilter->SetGamma(gamma);
  objectnessFilter->SetObjectDimension(objectDimension);
  objectnessFilter->SetScaleObjectnessMeasure(scaleObjectnessMeasure);

  multiScaleEnhancementFilter->Update();

  typedef HessianValueType::EigenValuesArrayType EigenValuesArrayType;
  typedef HessianValueType::EigenVectorsMatrixType EigenVectorsMatrixType;
  typedef itk::SymmetricEigenAnalysis<HessianValueType, EigenValuesArrayType, EigenVectorsMatrixType> EigenAnalysisType;

  EigenAnalysisType eigenAnalysis;

  eigenAnalysis.SetDimension(NumDimensions);
  eigenAnalysis.SetOrderEigenMagnitudes(true);

  constexpr unsigned int OutputVectorDimension = 1 + NumDimensions + 1; // each output value consists of measure(1), eigenVector(n) and scale(1)

  typedef itk::Vector<ValueType, OutputVectorDimension> OutputVectorType;
  typedef itk::Image<OutputVectorType, NumDimensions> OutputImageType;

  typedef itk::ImageFileWriter<OutputImageType> FileWriterType;

  OutputImageType::Pointer outputImage =
    OutputImageType::New();

  typedef itk::ImageRegionIterator<OutputImageType> OutputImageIteratorType;

  outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
  outputImage->Allocate();

  typedef itk::RescaleIntensityImageFilter<ImageType> RescaleIntensityFilterType;

  RescaleIntensityFilterType::Pointer rescaleIntensityFilter =
    RescaleIntensityFilterType::New();

  rescaleIntensityFilter->SetInput(multiScaleEnhancementFilter->GetOutput());
  rescaleIntensityFilter->SetOutputMinimum(outputMinimum);
  rescaleIntensityFilter->SetOutputMaximum(outputMaximum);
  rescaleIntensityFilter->Update();

  OutputImageIteratorType it(outputImage, outputImage->GetLargestPossibleRegion());

  ImageType::ConstPointer measureImage =
    rescaleIntensityFilter->GetOutput();

  HessianImageType::ConstPointer hessianImage =
    multiScaleEnhancementFilter->GetHessianOutput();

  ImageType::ConstPointer scalesImage =
    multiScaleEnhancementFilter->GetScalesOutput();

  OutputVectorType outputVector;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const IndexType index = it.GetIndex();

    if (inputImage->GetPixel(index) < thresholdBelow)
    {
      outputVector.Fill(0);
      outputVector.SetElement(1, 1);
    }
    else
    {
      EigenValuesArrayType eigenValues;
      EigenVectorsMatrixType eigenVectors;

      eigenAnalysis.ComputeEigenValuesAndVectors(hessianImage->GetPixel(index), eigenValues, eigenVectors);

      outputVector.SetElement(0, measureImage->GetPixel(index));

      for (int d = 0; d < NumDimensions; ++d)
      {
        outputVector.SetElement(d + 1, eigenVectors(0, d));
      }

      outputVector.SetElement(NumDimensions + 1, scalesImage->GetPixel(index));
    }

    it.Set(outputVector);
  }

  typedef itk::MetaDataDictionary MetaDataDictionaryType;

  MetaDataDictionaryType outputMetaData;

  EncapsulateMetaData(outputMetaData, "(ThresholdBelow)", thresholdBelow);
  EncapsulateMetaData(outputMetaData, "(SigmaMaximum)", sigmaMaximum);
  EncapsulateMetaData(outputMetaData, "(SigmaMinimum)", sigmaMinimum);
  EncapsulateMetaData(outputMetaData, "(NumberOfSigmaSteps)", numberOfSigmaSteps);
  EncapsulateMetaData(outputMetaData, "(Alpha)", alpha);
  EncapsulateMetaData(outputMetaData, "(Beta)", beta);
  EncapsulateMetaData(outputMetaData, "(Gamma)", gamma);
  EncapsulateMetaData(outputMetaData, "(ObjectDimension)", objectDimension);
  EncapsulateMetaData(outputMetaData, "(ScaleObjectnessMeasure)", scaleObjectnessMeasure);
  EncapsulateMetaData(outputMetaData, "(OutputMaximum)", outputMaximum);
  EncapsulateMetaData(outputMetaData, "(OutputMinimum)", outputMinimum);

  outputImage->SetMetaDataDictionary(outputMetaData);

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

  double thresholdBelow = 0;

  double sigmaMaximum = 5.0;
  double sigmaMinimum = 0.5;
  unsigned int numberOfSigmaSteps = 50;

  double alpha = 0.5;
  double beta = 0.5;
  double gamma = 25.0;
  unsigned int objectDimension = 1;
  bool scaleObjectnessMeasure = true;

  double outputMaximum = 1.0;
  double outputMinimum = 0.0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("thresholdBelow", po::value(&thresholdBelow), "the values below the threshold will be ignored")
    ("sigmaMaximum", po::value(&sigmaMaximum), "the minimum sigma value")
    ("sigmaMinimum", po::value(&sigmaMinimum), "the maximum sigma value")
    ("numberOfSigmaSteps", po::value(&numberOfSigmaSteps), "the number of scale levels")
    ("alpha", po::value(&alpha), "the weight corresponding to R_A (the ratio of the smallest eigenvalue that has to be large to the larger ones). Smaller values lead to increased sensitivity to the object dimensionality")
    ("beta", po::value(&beta), "the weight corresponding to R_B (the ratio of the largest eigenvalue that has to be small to the larger ones). Smaller values lead to increased sensitivity to the object dimensionality")
    ("gamma", po::value(&gamma), "the weight corresponding to S (the Frobenius norm of the Hessian matrix, or second-order structureness)")
    ("objectDimension", po::value(&objectDimension), "the dimensionality of the object (0: points (blobs), 1: lines (vessels), 2: planes (plate-like structures), 3: hyper-planes")
    ("scaleObjectnessMeasure", po::value(&scaleObjectnessMeasure), "scaling the objectness measure with the magnitude of the largest absolute eigenvalue")
    ("outputMaximum", po::value(&outputMaximum), "the maximum value that the output image should have")
    ("outputMinimum", po::value(&outputMinimum), "the minimum value that the output image should have")
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
    DoObjectnessMeasureImageFilter(inputFileName, outputFileName, thresholdBelow, sigmaMaximum, sigmaMinimum, numberOfSigmaSteps, alpha, beta, gamma, objectDimension, scaleObjectnessMeasure, outputMaximum, outputMinimum);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
