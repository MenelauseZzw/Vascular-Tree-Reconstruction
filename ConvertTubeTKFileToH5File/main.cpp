#include "FileWriter.hpp"
#include <boost/program_options.hpp>
#include <boost/log/trivial.hpp>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorImage.h>
#include <itkVectorImageToImageAdaptor.h>
#include <itkVesselTubeSpatialObject.h>
#include <itkVesselTubeSpatialObjectPoint.h>
#include <itkSpatialObject.h>
#include <itkSpatialObjectReader.h>
#include <iostream>
#include <string>

void DoConvertTubeTKFileToH5File(
  const std::string& inputFileName,
  const std::string& outputFileName)
{
  constexpr unsigned int NumDimensions = 3;
  typedef double ValueType;

  const std::string positionsDataSetName = "positions";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";

  typedef itk::VesselTubeSpatialObject<NumDimensions> VesselTubeSpatialObjectType;
  typedef itk::VesselTubeSpatialObjectPoint<NumDimensions> VesselTubeSpatialObjectPointType;
  typedef itk::SpatialObject<NumDimensions> SpatialObjectType;
  typedef itk::SpatialObjectReader<NumDimensions> SpatialObjectReaderType;
  typedef VesselTubeSpatialObjectType::TransformType TransformType;

  SpatialObjectReaderType::Pointer spatialObjectReader =
    SpatialObjectReaderType::New();

  spatialObjectReader->SetFileName(inputFileName);
  spatialObjectReader->Update();

  SpatialObjectReaderType::ScenePointer scene =
    spatialObjectReader->GetScene();

  std::vector<ValueType> positions;
  std::vector<ValueType> tangentLinesPoints1;
  std::vector<ValueType> tangentLinesPoints2;
  std::vector<ValueType> radiuses;

  for (SpatialObjectType* spatialObject : *scene->GetObjects())
  {
    if (VesselTubeSpatialObjectType* tubeObject = dynamic_cast<VesselTubeSpatialObjectType*>(spatialObject))
    {
      tubeObject->ComputeObjectToWorldTransform();

      for (VesselTubeSpatialObjectPointType const& objectPoint : tubeObject->GetPoints())
      {
        TransformType::Pointer indexToWorldTransform =
          tubeObject->GetIndexToWorldTransform();

        const auto p = indexToWorldTransform->TransformPoint(objectPoint.GetPosition());
        const auto s = indexToWorldTransform->TransformPoint(objectPoint.GetPosition() + objectPoint.GetTangent());
        const auto t = indexToWorldTransform->TransformPoint(objectPoint.GetPosition() - objectPoint.GetTangent());
        const ValueType r = indexToWorldTransform->TransformVector(objectPoint.GetRadius()).GetElement(0);

        positions.insert(positions.end(), p.Begin(), p.End());
        tangentLinesPoints1.insert(tangentLinesPoints1.end(), s.Begin(), s.End());
        tangentLinesPoints2.insert(tangentLinesPoints2.end(), t.Begin(), t.End());
        radiuses.push_back(r);
      }
    }
  }

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(positionsDataSetName, positions);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
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
    DoConvertTubeTKFileToH5File(inputFileName, outputFileName);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
