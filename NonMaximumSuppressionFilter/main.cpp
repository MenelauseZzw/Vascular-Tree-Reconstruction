#include <boost/program_options.hpp>
#include <iostream>
#include <cmath>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkIndex.h>
#include <itkMacro.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkOffset.h>
#include <itkVectorImage.h>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <queue>
#include <vector>

template<unsigned int NumDimensions, typename OffsetType = itk::Offset<NumDimensions>, typename OffsetValueType = typename OffsetType::OffsetValueType>
void InitializeCubeEdgesEndPoints(std::vector<itk::Offset<NumDimensions>>& offsetsOfEndPoint1, std::vector<itk::Offset<NumDimensions>>& offsetsOfEndPoint2, OffsetValueType minOffsetValue = -1, OffsetValueType maxOffsetValue = 1)
{
  OffsetType startingOffset;
  startingOffset.Fill(minOffsetValue);

  std::queue<OffsetType> Q;
  std::set<OffsetType, typename OffsetType::LexicographicCompare> S;

  Q.push(startingOffset);

  OffsetType zeroOffset;
  zeroOffset.Fill(0);

  assert(startingOffset != zeroOffset);

  while (!Q.empty())
  {
    const OffsetType offsetOfEndPoint1 = Q.front();
    Q.pop();

    if (!S.count(offsetOfEndPoint1))
    {
      S.insert(offsetOfEndPoint1);

      for (unsigned int i = 0; i < NumDimensions; ++i)
        if (offsetOfEndPoint1[i] < maxOffsetValue)
        {
          const OffsetType offsetOfEndPoint2 = offsetOfEndPoint1 + OffsetType::GetBasisOffset(i);

          if (offsetOfEndPoint2 != zeroOffset)
          {
            offsetsOfEndPoint1.push_back(offsetOfEndPoint1);
            offsetsOfEndPoint2.push_back(offsetOfEndPoint2);
            Q.push(offsetOfEndPoint2);
          }
        }
    }
  }
}

enum IntersectionType
{
  NoIntersection, PointIntersection, LineIntersection
};

template<typename ValueType, unsigned int NumDimensions>
ValueType DotProduct(const ValueType(&s)[NumDimensions], const ValueType(&t)[NumDimensions])
{
  return std::inner_product(std::begin(s), std::end(s), std::begin(t), ValueType());
}

template<unsigned int NumDimensions, typename ValueType, typename IndexType = itk::Index<NumDimensions>>
IntersectionType ComputeLineIntervalPlaneIntersection(const itk::Index<NumDimensions>& s, const IndexType& t, const IndexType& p0, const ValueType(&n)[NumDimensions], ValueType sVal, ValueType tVal, ValueType(&p)[NumDimensions], ValueType& pVal, ValueType eps = 0.01)
{
  ValueType sMinusP0[NumDimensions];
  ValueType sMinusT[NumDimensions];

  for (unsigned int i = 0; i < NumDimensions; ++i)
  {
    sMinusP0[i] = s[i] - p0[i];
    sMinusT[i] = s[i] - t[i];
  }

  const ValueType numLambda = DotProduct(sMinusP0, n);
  const ValueType denLambda = DotProduct(sMinusT, n);

  if (std::abs(denLambda) < eps) //then the line and plane are parallel
  {
    if (std::abs(numLambda) < eps) //then the line is contained in the plane
    {
      return LineIntersection;
    }
    else
    {
      return NoIntersection;
    }
  }
  else
  {
    const ValueType lambda = numLambda / denLambda;

    if (lambda < 0 || lambda > 1)
    {
      return NoIntersection;
    }
    else
    {
      const ValueType lambda = numLambda / denLambda;

      assert(lambda >= 0);
      assert(lambda <= 1);

      for (unsigned int i = 0; i < NumDimensions; ++i)
      {
        p[i] = s[i] - lambda * sMinusT[i];
      }

      pVal = (1 - lambda) * sVal + lambda * tVal;

      assert(pVal >= std::min(sVal, tVal));
      assert(pVal <= std::max(sVal, tVal));

      return PointIntersection;
    }
  }
}

template<unsigned int NumDimensions, typename ValueType, typename IndexType = itk::Index<NumDimensions>>
bool LineIntervalIntersectsPlane(const itk::Index<NumDimensions>& indexOfCenter, const ValueType(&normalVector)[NumDimensions], const IndexType& indexOfPoint1, const IndexType& indexOfPoint2, ValueType valueAtPoint1, ValueType valueAtPoint2, ValueType& valueAtXing)
{
  ValueType intersPoint[NumDimensions];

  IntersectionType intersection = ComputeLineIntervalPlaneIntersection(indexOfPoint1, indexOfPoint2, indexOfCenter, normalVector, valueAtPoint1, valueAtPoint2, intersPoint, valueAtXing);

  if (intersection == PointIntersection)
  {
    return true;
  }
  else if (intersection == LineIntersection)
  {
    valueAtXing = std::max(valueAtPoint1, valueAtPoint2);
    return true;
  }
  else return false;
}

void DoNonMaximumSuppressionFilter(const std::string& inputFileName, const std::string& outputFileName, double thresholdBelow)
{
  typedef float ValueType;

  constexpr unsigned int NumDimensions = 3;

  constexpr unsigned int ObjectnessMeasureValueComponentIndex = 0;
  constexpr unsigned int ObjectnessMeasureTangentsComponentIndex = 1;
  constexpr unsigned int SigmaValueComponentIndex = 1 + NumDimensions;

  constexpr unsigned int VectorDimension = 1 + NumDimensions + 1; // each output value consists of measure(1), eigenVector(n) and scale(1)

  typedef itk::Vector<ValueType, VectorDimension> VectorType;
  typedef itk::Image<VectorType, NumDimensions> ImageType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef itk::Offset<NumDimensions> OffsetType;
  typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;

  typedef itk::ImageFileReader<ImageType> FileReaderType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ImageType::ConstPointer inputImage =
    imageReader->GetOutput();

  typedef itk::ImageFileWriter<ImageType> FileWriterType;

  ImageType::Pointer outputImage =
    ImageType::New();

  outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
  outputImage->Allocate();

  ImageIteratorType it(outputImage, outputImage->GetLargestPossibleRegion());

  VectorType zeroVector;
  zeroVector.Fill(0);

  std::vector<OffsetType> offsetsOfEndPoint1;
  std::vector<OffsetType> offsetsOfEndPoint2;

  InitializeCubeEdgesEndPoints(offsetsOfEndPoint1, offsetsOfEndPoint2);
  assert(offsetsOfEndPoint1.size() == offsetsOfEndPoint2.size());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    const IndexType indexOfCenter = it.GetIndex();
    const ValueType objectnessMeasureValueAtCenter = inputImage->GetPixel(indexOfCenter).GetElement(ObjectnessMeasureValueComponentIndex);

    if (objectnessMeasureValueAtCenter < thresholdBelow)
    {
      it.Set(zeroVector);
      continue;
    }

    ValueType objectnessMeasureTangentAtCenter[NumDimensions];

    for (unsigned int k = 0; k < NumDimensions; ++k)
    {
      objectnessMeasureTangentAtCenter[k] = inputImage->GetPixel(indexOfCenter).GetElement(ObjectnessMeasureTangentsComponentIndex + k);
    }

    bool isValueAtCenterLocalMaximum = true;

    for (unsigned int i = 0; i < offsetsOfEndPoint1.size(); ++i)
    {
      const OffsetType& offsetOfEndPoint1 = offsetsOfEndPoint1[i];
      const OffsetType& offsetOfEndPoint2 = offsetsOfEndPoint2[i];

      const IndexType indexOfEndPoint1 = indexOfCenter + offsetOfEndPoint1;

      if (outputImage->GetLargestPossibleRegion().IsInside(indexOfEndPoint1))
      {
        const IndexType indexOfEndPoint2 = indexOfCenter + offsetOfEndPoint2;

        if (outputImage->GetLargestPossibleRegion().IsInside(indexOfEndPoint2))
        {
          const ValueType valueAtPoint1 = inputImage->GetPixel(indexOfEndPoint1).GetElement(ObjectnessMeasureValueComponentIndex);
          const ValueType valueAtPoint2 = inputImage->GetPixel(indexOfEndPoint2).GetElement(ObjectnessMeasureValueComponentIndex);

          ValueType valueAtXing;

          if (LineIntervalIntersectsPlane(indexOfCenter, objectnessMeasureTangentAtCenter, indexOfEndPoint1, indexOfEndPoint2, valueAtPoint1, valueAtPoint2, valueAtXing))
          {
            if (objectnessMeasureValueAtCenter <= valueAtXing)
            {
              isValueAtCenterLocalMaximum = false;
              break;
            }
          }
        }
      }
    }

    if (isValueAtCenterLocalMaximum)
      it.Set(inputImage->GetPixel(indexOfCenter));
    else
      it.Set(zeroVector);
  }

  typedef itk::MetaDataDictionary MetaDataDictionaryType;

  MetaDataDictionaryType outMetaData;

  EncapsulateMetaData(outMetaData, "(ThresholdBelow)", thresholdBelow);

  outputImage->SetOrigin(inputImage->GetOrigin());
  outputImage->SetSpacing(inputImage->GetSpacing());

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

  double thresholdBelow = 0.01;

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
    DoNonMaximumSuppressionFilter(inputFileName, outputFileName, thresholdBelow);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
