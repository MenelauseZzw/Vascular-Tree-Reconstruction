#include <algorithm>
#include <array>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <itk_H5Cpp.h>
#include <iostream>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageSeriesReader.h>
#include <itkImageToVTKImageFilter.h>
#include <itkRawImageIO.h>
#include <itkNumericSeriesFileNames.h>
#include "pugixml.hpp"
#include <map>
#include <string>
#include <vtkGraphToPolyData.h>
#include <vtkLine.h>
#include <vtkMetaImageWriter.h>
#include <vtkMutableDirectedGraph.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkAdjacentVertexIterator.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

DEFINE_string(source, "", "source filename");
//DEFINE_string(source2, "C:\\Temp\\VascuSynth.tmp\\Debug\\image1\\original_image\\image%03d.jpg", "source filename");

//DEFINE_string(result1, "C:\\WesternU\\CompareWithVascuSynthDataset\\Dataset2.SourceGraph.vtp", "result filename");
//DEFINE_string(result2, "C:\\WesternU\\CompareWithVascuSynthDataset\\Dataset2.data", "result filename");
//DEFINE_string(result3, "C:\\WesternU\\CompareWithVascuSynthDataset\\Dataset2.SourceImage.mhd", "result filename");
DEFINE_string(result, "", "result filename");

const int Dimension = 3;

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

template<typename ValueType>
PredType GetDataType();

template<>
PredType GetDataType<float>() { return PredType::NATIVE_FLOAT; }

template<>
PredType GetDataType<double>() { return PredType::NATIVE_DOUBLE; }

template<>
PredType GetDataType<int>() { return PredType::NATIVE_INT; }

template<typename ValueType>
std::vector<ValueType> Read(const H5File& sourceFile, const std::string& sourceDataSetName)
{
  auto sourceDataSet = sourceFile.openDataSet(sourceDataSetName);
  vector<ValueType> resultDataSet(sourceDataSet.getSpace().getSimpleExtentNpoints());
  sourceDataSet.read(&resultDataSet[0], GetDataType<ValueType>());
  return resultDataSet;
}

template<typename ValueType>
void Write(H5File& resultFile, const std::string& resultDataSetName, const std::vector<ValueType>& sourceDataSet)
{
  const int rank = 1;
  const hsize_t dims = sourceDataSet.size();

  auto resultDataSet = resultFile.createDataSet(resultDataSetName, GetDataType<ValueType>(), DataSpace(rank, &dims));
  resultDataSet.write(&sourceDataSet[0], GetDataType<ValueType>());
}

int main(int argc, char *argv[])
{
  const int NumDimensions{ 3 };
  typedef int IndexType;
  typedef double ValueType;

  using namespace std;

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  {
    auto const& sourceFilename = FLAGS_source;/*
    auto const& resultFilename = FLAGS_result1;*/

    pugi::xml_document source;
    LOG_ASSERT(source.load_file(sourceFilename.c_str()));

    typedef std::array<float, Dimension> PositionType;

    auto graph = vtkSmartPointer<vtkMutableDirectedGraph>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    std::map<std::string, vtkIdType> vtkIds;

    for (auto const& node : source.select_nodes("/gxl/graph/node"))
    {
      std::string id = node.node().attribute("id").as_string();

      std::string nodeType = node.node().select_single_node("attr[normalize-space(@name) = 'nodeType']/string").node().child_value();

      std::string::size_type begin = nodeType.find_first_not_of(' ');
      std::string::size_type end = nodeType.find_last_not_of(' ');

      nodeType.erase(0, begin);
      nodeType.erase(end);

      PositionType position;

      auto const& coordinates = node.node().select_nodes("attr[normalize-space(@name) = 'position']/tup/float");
      std::transform(std::begin(coordinates), std::end(coordinates), std::begin(position), [](pugi::xpath_node const& coordinate)
      {
        return std::stof(coordinate.node().child_value());
      });

      points->InsertNextPoint(position[0], position[1], position[2]);
      vtkIdType vtkId = graph->AddVertex();
      vtkIds.insert({ id, vtkId });
    }

    // Create a polydata object and add the points to it.
    //auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    //grid->SetPoints(points);

    graph->SetPoints(points);
    std::vector<ValueType> radiuses;
    std::vector<ValueType> radiusesPrime;

    for (auto const& node : source.select_nodes("/gxl/graph/edge"))
    {
      std::string id = node.node().attribute("id").as_string();
      std::string to = node.node().attribute("to").as_string();
      std::string from = node.node().attribute("from").as_string();

      float flow = std::stof(node.node().select_single_node("attr[normalize-space(@name) = 'flow']/float").node().child_value());
      float radius = 20 * std::stof(node.node().select_single_node("attr[normalize-space(@name) = 'radius']/float").node().child_value());

      //vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();

      //line->GetPointIds()->SetId(0, vtkIds[from]);
      //line->GetPointIds()->SetId(1, vtkIds[to]);

      //grid->InsertNextCell(line->GetCellType(), line->GetPointIds());

      graph->AddEdge(vtkIds[from], vtkIds[to]);
      radiuses.push_back(radius);
    }

    //auto graphToPolyData = vtkSmartPointer<vtkGraphToPolyData>::New();
    //graphToPolyData->SetInputData(graph);

    // Write the file
    //auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    //writer->SetFileName(resultFilename.c_str());
    //writer->SetInputConnection(graphToPolyData->GetOutputPort());
    //writer->Update();
    //writer->Write();

    vector<ValueType> measurements;
    vector<ValueType> tangentLinesPoints1;
    vector<ValueType> tangentLinesPoints2;
    vector<ValueType> radiuses_;

    vector<IndexType> indices1;
    vector<IndexType> indices2;

    for (int i = 0; i < graph->GetNumberOfVertices(); ++i)
    {
      double point[NumDimensions];
      graph->GetPoint(i, point);

      for (int k = 0; k < NumDimensions; ++k)
      {
        measurements.push_back(point[k]);
      }

      auto iterator = vtkSmartPointer<vtkAdjacentVertexIterator>::New();
      graph->GetAdjacentVertices(i, iterator);

      int adjacentWithMaxRadius = -1;
      double maxRadius = 0;
      for (; iterator->HasNext();)
      {
        int k = iterator->Next();

        indices1.push_back(i);
        indices2.push_back(k);

        ValueType radius = radiuses[graph->GetEdgeId(i, k)];
        radiusesPrime.push_back(radius);

        if (adjacentWithMaxRadius == -1 || radius > maxRadius)
        {
          adjacentWithMaxRadius = k;
          maxRadius = radius;
        }
      }

      radiuses_.push_back(maxRadius);

      double adjacentPoint[NumDimensions];
      graph->GetPoint(adjacentWithMaxRadius, adjacentPoint);
      
      double sMinusTSq = 0;
      double sMinusT[NumDimensions];
      for (int k = 0; k < NumDimensions; ++k)
      {
        sMinusT[k] = point[k] - adjacentPoint[k];
        sMinusTSq += sMinusT[k] * sMinusT[k];
      }

      for (int k = 0; k < NumDimensions; ++k)
      {
        sMinusT[k] /= sqrt(sMinusTSq);
      }

      for (int k = 0; k < NumDimensions; ++k)
      {
        tangentLinesPoints1.push_back(point[k] + sMinusT[k]);
        tangentLinesPoints2.push_back(point[k] - sMinusT[k]);
      }
    }

    const string measurementsDataSetName{ "measurements" };
    const string tangentLinesPoints1DataSetName{ "tangentLinesPoints1" };
    const string tangentLinesPoints2DataSetName{ "tangentLinesPoints2" };
    const string positionsDataSetName{ "positions" };
    const string radiusesDataSetName{ "radiuses" };
    const string indices1DataSetName{ "indices1" };
    const string indices2DataSetName{ "indices2" };
    const string radiusesPrimeDataSetName{ "radiuses'" };

    H5File resultFile2(FLAGS_result, H5F_ACC_TRUNC);

    Write(resultFile2, measurementsDataSetName, measurements);
    Write(resultFile2, tangentLinesPoints1DataSetName, tangentLinesPoints1);
    Write(resultFile2, tangentLinesPoints2DataSetName, tangentLinesPoints2);
    Write(resultFile2, positionsDataSetName, measurements);
    Write(resultFile2, radiusesDataSetName, radiuses_);
    Write(resultFile2, radiusesPrimeDataSetName, radiusesPrime);
    Write(resultFile2, indices1DataSetName, indices1);
    Write(resultFile2, indices2DataSetName, indices2);
  }

  /*{
    typedef short PixelType;
    typedef itk::Image<PixelType, Dimension> InputImageType;

    typedef itk::Image<PixelType, Dimension> InputImageType;

    typedef itk::ImageSeriesReader<InputImageType> ReaderType;
    typedef itk::ImageToVTKImageFilter<InputImageType> ConnectorType;

    ReaderType::Pointer reader = ReaderType::New();
    ConnectorType::Pointer connector = ConnectorType::New();

    typedef itk::NumericSeriesFileNames NameGeneratorType;
    NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();

    auto const& seriesFormat = FLAGS_source2;
    auto const& resultFilename = FLAGS_result2;

    nameGenerator->SetSeriesFormat(seriesFormat.c_str());
    nameGenerator->SetStartIndex(0);
    nameGenerator->SetEndIndex(100);
    nameGenerator->SetIncrementIndex(1);

    reader->SetFileNames(nameGenerator->GetFileNames());

    connector->SetInput(reader->GetOutput());
    connector->Update();

    typedef itk::RawImageIO<PixelType, Dimension> RawImageIOType;
    typedef itk::ImageFileWriter<InputImageType> WriterType;

    auto imageIO = RawImageIOType::New();
    imageIO->SetByteOrderToLittleEndian();

    auto writer = WriterType::New();
    writer->SetImageIO(imageIO);
    writer->SetFileName(resultFilename.c_str());
    writer->SetInput(reader->GetOutput());
    writer->Write();
  }*/

  /*{
    typedef short PixelType;
    typedef itk::Image<PixelType, Dimension> InputImageType;

    typedef itk::Image<PixelType, Dimension> InputImageType;

    typedef itk::ImageSeriesReader<InputImageType> ReaderType;
    typedef itk::ImageToVTKImageFilter<InputImageType> ConnectorType;

    ReaderType::Pointer reader = ReaderType::New();
    ConnectorType::Pointer connector = ConnectorType::New();

    typedef itk::NumericSeriesFileNames NameGeneratorType;
    NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();

    auto const& seriesFormat = FLAGS_source2;
    auto const& resultFilename = FLAGS_result3;

    nameGenerator->SetSeriesFormat(seriesFormat.c_str());
    nameGenerator->SetStartIndex(0);
    nameGenerator->SetEndIndex(100);
    nameGenerator->SetIncrementIndex(1);

    reader->SetFileNames(nameGenerator->GetFileNames());

    connector->SetInput(reader->GetOutput());
    connector->Update();

    typedef itk::ImageFileWriter<InputImageType> WriterType;

    auto writer = WriterType::New();
    writer->SetFileName(resultFilename.c_str());
    writer->SetInput(reader->GetOutput());
    writer->Write();
  }*/
}