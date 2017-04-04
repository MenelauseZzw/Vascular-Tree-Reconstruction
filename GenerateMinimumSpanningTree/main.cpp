#include "FileReader.hpp"
#include "FileWriter.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

template<typename ValueType, typename PositionType>
ValueType ComputeEuclideanDistance(
  const PositionType& node1,
  const PositionType& node2,
  const PositionType& tangentLine1Point1,
  const PositionType& tangentLine1Point2,
  const PositionType& tangentLine2Point1,
  const PositionType& tangentLine2Point2,
  ValueType radius1,
  ValueType radius2)
{
  return (node1 - node2).norm();
}

template<int NumDimensions, typename ValueType, typename IndexType, typename PositionType>
void GenerateMinimumSpanningTreeWith(
  const std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType, ValueType)>& distanceFunc,
  const std::vector<ValueType>& positions,
  const std::vector<ValueType>& tangentLinesPoints1,
  const std::vector<ValueType>& tangentLinesPoints2,
  const std::vector<ValueType>& radiuses,
  std::vector<IndexType>& indices1,
  std::vector<IndexType>& indices2)
{
  using namespace std;
  using namespace boost;
  using namespace Eigen;

  typedef adjacency_list<vecS, vecS, undirectedS, no_property, property<edge_weight_t, ValueType>> GraphType;
  typedef typename graph_traits<GraphType>::edge_descriptor GraphEdgeType;
  typedef typename property_map<GraphType, edge_weight_t>::type WeightMapType;

  typedef Matrix<ValueType, Dynamic, NumDimensions, RowMajor> MatrixType;

  const size_t numberOfPoints = positions.size() / NumDimensions;

  Map<const MatrixType> positions_(positions.data(), numberOfPoints, NumDimensions);
  Map<const MatrixType> tangentLinesPoints1_(tangentLinesPoints1.data(), numberOfPoints, NumDimensions);
  Map<const MatrixType> tangentLinesPoints2_(tangentLinesPoints2.data(), numberOfPoints, NumDimensions);

  GraphType completeGraph(numberOfPoints);
  WeightMapType weightmap = get(edge_weight, completeGraph);

  GraphEdgeType e;
  for (IndexType index1 = 0; index1 < numberOfPoints; ++index1)
  {
    for (IndexType index2 = index1 + 1; index2 < numberOfPoints; ++index2)
    {
      tie(e, tuples::ignore) = add_edge(index1, index2, completeGraph);

      const auto& node1 = positions_.row(index1);
      const auto& tangentLine1Point1 = tangentLinesPoints1_.row(index1);
      const auto& tangentLine1Point2 = tangentLinesPoints2_.row(index1);
      const ValueType radius1 = radiuses[index1];

      const auto& node2 = positions_.row(index2);
      const auto& tangentLine2Point1 = tangentLinesPoints1_.row(index2);
      const auto& tangentLine2Point2 = tangentLinesPoints2_.row(index2);
      const ValueType radius2 = radiuses[index2];

      const ValueType distance = distanceFunc(node1, node2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2);
      weightmap[e] = distance;
    }
  }

  std::vector<GraphEdgeType> spanningTree;
  spanningTree.reserve(numberOfPoints - 1);

  kruskal_minimum_spanning_tree(completeGraph, std::back_inserter(spanningTree));

  for (const GraphEdgeType& e : spanningTree)
  {
    const IndexType index1 = source(e, completeGraph);
    const IndexType index2 = target(e, completeGraph);

    indices1.push_back(index1);
    indices2.push_back(index2);
  }
}

void DoGenerateMinimumSpanningTree(const std::string& inputFileName, const std::string& outputFileName)
{
  const int NumDimensions = 3;

  typedef double ValueType;
  typedef int IndexType;

  typedef Eigen::Matrix<ValueType, 1, NumDimensions> PositionType;
  typedef std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType, ValueType)> DistanceFunctionType;

  const std::string measurementsDataSetName = "measurements";
  const std::string positionsDataSetName = "positions";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";
  const std::string objectnessMeasureDataSetName = "objectnessMeasure";

  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";

  FileReader inputFileReader(inputFileName);

  std::vector<ValueType> measurements;
  std::vector<ValueType> positions;
  std::vector<ValueType> tangentLinesPoints1;
  std::vector<ValueType> tangentLinesPoints2;
  std::vector<ValueType> radiuses;
  std::vector<ValueType> objectnessMeasure;

  inputFileReader.Read(measurementsDataSetName, measurements);
  inputFileReader.Read(positionsDataSetName, positions);
  inputFileReader.Read(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  inputFileReader.Read(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  inputFileReader.Read(radiusesDataSetName, radiuses);
  inputFileReader.Read(objectnessMeasureDataSetName, objectnessMeasure);

  std::vector<IndexType> indices1;
  std::vector<IndexType> indices2;

  GenerateMinimumSpanningTreeWith<NumDimensions>(
    (DistanceFunctionType)ComputeEuclideanDistance<ValueType, PositionType>,
    positions,
    tangentLinesPoints1,
    tangentLinesPoints2,
    radiuses,
    indices1,
    indices2);

  BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
  BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(measurementsDataSetName, measurements);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
  outputFileWriter.Write(objectnessMeasureDataSetName, objectnessMeasure);
  outputFileWriter.Write(positionsDataSetName, positions);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

int main(int argc, char *argv[])
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
    DoGenerateMinimumSpanningTree(inputFileName, outputFileName);
    return EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
