// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "FileWriter.hpp"
#include <algorithm>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <functional>
#include <itkContinuousIndex.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIndex.h>
#include <itkPoint.h>
#include <itkSize.h>
#include <itkShapedNeighborhoodIterator.h>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

void DoGenerateNeighborhoodGraph(
  const std::string& inputFileName,
  const std::string& outputFileName,
  float thresholdBelow)
{
  using namespace boost;

  typedef float InputValueType;
  typedef double OutputValueType;
  typedef int OutputIndexType;

  constexpr unsigned int NumDimensions = 3;
  constexpr unsigned int VectorDimension = 1 + NumDimensions + 1; // each value consists of measure(1), eigenVector(n) and scale(1)

  constexpr unsigned int ObjectnessMeasureValueComponentIndex = 0;
  constexpr unsigned int ObjectnessMeasureTangentsComponentIndex = 1;
  constexpr unsigned int SigmaValueComponentIndex = 1 + NumDimensions;

  typedef itk::Vector<InputValueType, VectorDimension> PixelType;
  typedef itk::Image<PixelType, NumDimensions> ObjectnessMeasureImageType;

  typedef itk::Size<NumDimensions> SizeType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef itk::ContinuousIndex<OutputValueType, NumDimensions> ContinuousIndexType;
  typedef itk::Point<OutputValueType, NumDimensions> PointType;

  typedef typename SizeType::SizeValueType SizeValueType;

  typedef itk::ShapedNeighborhoodIterator<ObjectnessMeasureImageType> ImageRegionIteratorType;
  typedef typename ImageRegionIteratorType::ConstIterator NeighborhoodIteratorType;

  typedef itk::ImageFileReader<ObjectnessMeasureImageType> FileReaderType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ObjectnessMeasureImageType::Pointer inputImage =
    imageReader->GetOutput();

  constexpr SizeValueType radiusOfNeighborhood = 1;

  SizeType neighborhoodRadius;
  neighborhoodRadius.Fill(radiusOfNeighborhood);

  ImageRegionIteratorType imageRegionIterator(neighborhoodRadius, inputImage, inputImage->GetLargestPossibleRegion());

  SizeValueType sizeOfNeighborhood = 1;

  for (unsigned int k = 0; k < NumDimensions; ++k)
  {
    sizeOfNeighborhood *= (radiusOfNeighborhood + 1 + radiusOfNeighborhood);
  }

  for (SizeValueType i = 0; i < sizeOfNeighborhood; ++i)
  {
    itk::Offset<NumDimensions> offsetOfNeighbor = imageRegionIterator.GetOffset(i);

    bool doActivateOffset = true;

    for (unsigned int k = 0; k < NumDimensions; ++k)
    {
      if (offsetOfNeighbor[k] < 0)
      {
        doActivateOffset = false;
        break;
      }
    }

    if (doActivateOffset)
    {
      imageRegionIterator.ActivateOffset(offsetOfNeighbor);
    }
  }

  imageRegionIterator.DeactivateOffset(imageRegionIterator.GetOffset(imageRegionIterator.GetCenterNeighborhoodIndex()));

  typedef std::map<IndexType, OutputIndexType, typename IndexType::LexicographicCompare> IndexToOutputIndexMap;
  typedef IndexToOutputIndexMap::iterator IndexToOutputIndexMapIterator;
  IndexToOutputIndexMap indexToOutputIndex;

  std::vector<OutputValueType> measurements;
  std::vector<OutputValueType> tangentLinesPoints1;
  std::vector<OutputValueType> tangentLinesPoints2;
  std::vector<OutputValueType> radiuses;
  std::vector<OutputValueType> objectnessMeasure;

  std::vector<OutputIndexType> indices1;
  std::vector<OutputIndexType> indices2;

  std::function<OutputIndexType(IndexType const&, PixelType const&)> getIndexOrAdd = [&](IndexType const& indexOfPixel, PixelType const& valueAtPixel)
  {
    IndexToOutputIndexMapIterator iteratorAtPixel = indexToOutputIndex.find(indexOfPixel);
    OutputIndexType index;

    if (iteratorAtPixel != indexToOutputIndex.end())
    {
      index = iteratorAtPixel->second;
    }
    else
    {
      index = objectnessMeasure.size();
      const OutputValueType objectnessMeasureValue = valueAtPixel.GetElement(ObjectnessMeasureValueComponentIndex);
      const OutputValueType sigmaValue = valueAtPixel.GetElement(SigmaValueComponentIndex);

      objectnessMeasure.push_back(objectnessMeasureValue);

      PointType pointAtPixel;
      inputImage->TransformIndexToPhysicalPoint(indexOfPixel, pointAtPixel);

      ContinuousIndexType indexOfTangentLinePoint1;
      ContinuousIndexType indexOfTangentLinePoint2;

      for (unsigned int k = 0; k < NumDimensions; ++k)
      {
        const OutputValueType objectnessMeasureTangent = valueAtPixel.GetElement(ObjectnessMeasureTangentsComponentIndex + k);

        indexOfTangentLinePoint1.SetElement(k, indexOfPixel.GetElement(k) - objectnessMeasureTangent);
        indexOfTangentLinePoint2.SetElement(k, indexOfPixel.GetElement(k) + objectnessMeasureTangent);
      }

      PointType tangentLinePoint1;
      inputImage->TransformContinuousIndexToPhysicalPoint(indexOfTangentLinePoint1, tangentLinePoint1);

      PointType tangentLinePoint2;
      inputImage->TransformContinuousIndexToPhysicalPoint(indexOfTangentLinePoint2, tangentLinePoint2);

      copy(pointAtPixel.GetDataPointer(), pointAtPixel.GetDataPointer() + NumDimensions, std::back_inserter(measurements));
      copy(tangentLinePoint1.GetDataPointer(), tangentLinePoint1.GetDataPointer() + NumDimensions, std::back_inserter(tangentLinesPoints1));
      copy(tangentLinePoint2.GetDataPointer(), tangentLinePoint2.GetDataPointer() + NumDimensions, std::back_inserter(tangentLinesPoints2));

      radiuses.push_back(sigmaValue);

      indexToOutputIndex.insert({ indexOfPixel, index });
    }

    return index;
  };

  for (imageRegionIterator.GoToBegin(); !imageRegionIterator.IsAtEnd(); ++imageRegionIterator)
  {
    const PixelType* pValueAtCenterPixel = imageRegionIterator.GetCenterPointer();
    const InputValueType objectnessMeasureValueAtCenter = pValueAtCenterPixel->GetElement(ObjectnessMeasureValueComponentIndex);

    if (objectnessMeasureValueAtCenter < thresholdBelow) continue;
    const IndexType indexOfCenterPixel = imageRegionIterator.GetIndex();
    OutputIndexType outputIndexOfCenterPixel;

    bool indexOfCenterPixelValid = false;

    for (NeighborhoodIteratorType neighborhoodIterator = imageRegionIterator.Begin(); !neighborhoodIterator.IsAtEnd(); ++neighborhoodIterator)
    {
      const IndexType indexOfNeighbor = indexOfCenterPixel + neighborhoodIterator.GetNeighborhoodOffset();

      if (!inputImage->GetLargestPossibleRegion().IsInside(indexOfNeighbor))
        continue;

      const PixelType valueAtNeighbor = neighborhoodIterator.Get();
      const InputValueType objectnessMeasureValueAtNeighbor = valueAtNeighbor.GetElement(ObjectnessMeasureValueComponentIndex);

      if (objectnessMeasureValueAtNeighbor < thresholdBelow) continue;
     
      if (!indexOfCenterPixelValid)
      {
        outputIndexOfCenterPixel = getIndexOrAdd(indexOfCenterPixel, *pValueAtCenterPixel);
        indexOfCenterPixelValid = true;
      }

      OutputIndexType outputIndexOfNeighbor = getIndexOrAdd(indexOfNeighbor, valueAtNeighbor);

      indices1.push_back(outputIndexOfCenterPixel);
      indices2.push_back(outputIndexOfNeighbor);
    }
  }

  indices1.reserve(indices1.size() + indices2.size());
  copy(indices2.cbegin(), indices2.cend(), std::back_inserter(indices1));

  indices2.reserve(indices1.size());
  copy(indices1.cbegin(), indices1.cbegin() + indices2.size(), std::back_inserter(indices2));

  BOOST_LOG_TRIVIAL(info) << "measurements.size = " << measurements.size();
  BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints1.size = " << tangentLinesPoints1.size();
  BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints2.size = " << tangentLinesPoints2.size();
  BOOST_LOG_TRIVIAL(info) << "radiuses.size = " << radiuses.size();
  BOOST_LOG_TRIVIAL(info) << "objectnessMeasure.size = " << objectnessMeasure.size();

  BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
  BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();

  const std::string measurementsDataSetName = "measurements";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";
  const std::string objectnessMeasureDataSetName = "objectnessMeasure";

  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(measurementsDataSetName, measurements);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
  outputFileWriter.Write(objectnessMeasureDataSetName, objectnessMeasure);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  std::string inputFileName;
  std::string outputFileName;

  double thresholdBelow = 0.05;
  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("thresholdBelow", po::value(&thresholdBelow), "the values below the threshold will be ignored")
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
    DoGenerateNeighborhoodGraph(inputFileName, outputFileName, thresholdBelow);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
