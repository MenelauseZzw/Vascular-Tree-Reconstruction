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
#include <flann/flann.hpp>
#include <iostream>
#include <string>
#include <vector>

template<typename ValueType, typename PositionType>
ValueType ComputeEuclideanDistance(
  const PositionType& point1,
  const PositionType& point2,
  const PositionType& tangentLine1Point1,
  const PositionType& tangentLine1Point2,
  const PositionType& tangentLine2Point1,
  const PositionType& tangentLine2Point2,
  ValueType radius1,
  ValueType radius2)
{
  return (point1 - point2).norm();
}

template<typename ValueType, typename PositionType>
void ComputeArcCenterAndRadius(const PositionType& point1, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& point2, PositionType& center, ValueType& radius)
{
  using namespace std;

  const PositionType tangentLine1 = tangentLine1Point2 - tangentLine1Point1;
  const PositionType baseLine = point2 - point1;
  const PositionType radialLine1 = baseLine * tangentLine1.squaredNorm() - tangentLine1 * tangentLine1.dot(baseLine);//radialLine1 = tangentLine1 x (baseLine x tangentLine1) = baseLine * ||tangentLine1||^2 - tangentLine1 * <tangentLine1, baseLine>
  const ValueType t = 0.5 * baseLine.squaredNorm() / baseLine.dot(radialLine1);//x(t) = point1 + t * radialLine1: radius = ||point1 - center(t)||^2 = ||point2 - center(t)||^2 => t = (1/2) * ||baseLine||^2 / <baseLine, radialLine1>

  center = point1 + t * radialLine1;
  radius = abs(t) * radialLine1.norm();
}

template<typename ValueType, typename PositionType>
ValueType ComputeArcLength(const PositionType& point1, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& point2)
{
  PositionType arcCenter;
  ValueType arcRadius;

  ComputeArcCenterAndRadius(point1, tangentLine1Point1, tangentLine1Point2, point2, arcCenter, arcRadius);
  return arcRadius * acos((point1 - arcCenter).dot(point2 - arcCenter) / (arcRadius * arcRadius));
}

template<typename ValueType, typename PositionType>
ValueType ComputeArcLengthsSumDistance(const PositionType& point1, const PositionType& point2, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& tangentLine2Point1, const PositionType& tangentLine2Point2, ValueType radius1, ValueType radius2)
{
  const ValueType arc1Length = ComputeArcLength<ValueType>(point1, tangentLine1Point1, tangentLine1Point2, point2);
  const ValueType arc2Length = ComputeArcLength<ValueType>(point2, tangentLine2Point1, tangentLine2Point2, point1);

  return arc1Length + arc2Length;
}

template<typename ValueType, typename PositionType>
ValueType ComputeArcLengthsMinDistance(const PositionType& point1, const PositionType& point2, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& tangentLine2Point1, const PositionType& tangentLine2Point2, ValueType radius1, ValueType radius2)
{
  const ValueType arc1Length = ComputeArcLength<ValueType>(point1, tangentLine1Point1, tangentLine1Point2, point2);
  const ValueType arc2Length = ComputeArcLength<ValueType>(point2, tangentLine2Point1, tangentLine2Point2, point1);

  return std::min(arc1Length, arc2Length);
}

template<int NumDimensions, typename ValueType, typename IndexType>
void GenerateEuclideanMinimumSpanningTree(const std::vector<ValueType>& positions, std::vector<IndexType>& indices1, std::vector<IndexType>& indices2, int knn)
{
  using namespace boost;
  typedef adjacency_list<vecS, vecS, undirectedS, no_property, property<edge_weight_t, ValueType>> GraphType;
  typedef typename graph_traits<GraphType>::edge_descriptor GraphEdgeType;

  const size_t numberOfPoints = positions.size() / NumDimensions;

  GraphType origGraph(numberOfPoints);
  auto weightmap = get(edge_weight, origGraph);

  if (knn != -1)
  {
    using namespace std;
    using namespace flann;

    const Matrix<ValueType> dataset(const_cast<ValueType*>(positions.data()), positions.size() / NumDimensions, NumDimensions);

    Index<L2<ValueType>> index(dataset, flann::KDTreeIndexParams(1));
    index.buildIndex();

    vector<IndexType> indicesData(dataset.rows * (knn + 1));
    Matrix<IndexType> indices(&indicesData[0], dataset.rows, knn + 1);

    vector<ValueType> distancesData(dataset.rows * (knn + 1));
    Matrix<ValueType> distances(&distancesData[0], dataset.rows, knn + 1);

    index.knnSearch(dataset, indices, distances, knn + 1, flann::SearchParams(-1));

    for (IndexType index1 = 0; index1 != dataset.rows; ++index1)
    {
      for (IndexType i = 0; i != knn + 1; ++i)
      {
        const IndexType index2 = indices[index1][i];
        if (index1 == index2) continue;
        if (index2 == -1) break;

        GraphEdgeType e;
        tie(e, tuples::ignore) = add_edge(index1, index2, origGraph);

        weightmap[e] = distances[index1][i];
      }
    }
  }
  else
  {
    using namespace std;
    using namespace Eigen;

    typedef Matrix<ValueType, Dynamic, NumDimensions, RowMajor> MatrixType;
    Map<const MatrixType> points(positions.data(), numberOfPoints, NumDimensions);

    for (IndexType index1 = 0; index1 < numberOfPoints; ++index1)
    {
      for (IndexType index2 = index1 + 1; index2 < numberOfPoints; ++index2)
      {
        GraphEdgeType e;
        tie(e, tuples::ignore) = add_edge(index1, index2, origGraph);

        const auto& point1 = points.row(index1);
        const auto& point2 = points.row(index2);
        const ValueType distance = (point1 - point2).norm();

        weightmap[e] = distance;
      }
    }
  }

  std::vector<GraphEdgeType> spanningTree;
  spanningTree.reserve(numberOfPoints - 1);

  kruskal_minimum_spanning_tree(origGraph, std::back_inserter(spanningTree));

  indices1.reserve(spanningTree.size());
  indices2.reserve(spanningTree.size());

  for (const GraphEdgeType& e : spanningTree)
  {
    const IndexType index1 = source(e, origGraph);
    const IndexType index2 = target(e, origGraph);

    indices1.push_back(index1);
    indices2.push_back(index2);
  }
}

template<typename ValueType, typename IndexType>
void GenerateKnnGraph(
  int knn,
  const std::vector<ValueType>& positions,
  const std::vector<ValueType>& tangentLinesPoints1,
  const std::vector<ValueType>& tangentLinesPoints2,
  std::vector<IndexType>& indices1,
  std::vector<IndexType>& indices2,
  int numDimensions)
{
  using namespace flann;
  using namespace std;

  const Matrix<ValueType> dataset(const_cast<ValueType*>(positions.data()), positions.size() / numDimensions, numDimensions);

  Index<L2<ValueType>> index(dataset, flann::KDTreeIndexParams(1));
  index.buildIndex();

  vector<IndexType> indicesData(dataset.rows * (knn + 1));
  Matrix<IndexType> indices(&indicesData[0], dataset.rows, knn + 1);

  vector<ValueType> distancesData(dataset.rows * (knn + 1));
  Matrix<ValueType> distances(&distancesData[0], dataset.rows, knn + 1);

  index.knnSearch(dataset, indices, distances, knn + 1, flann::SearchParams(-1));

  indices1.reserve(indices.rows * indices.cols);
  indices2.reserve(indices1.size());

  for (IndexType index1 = 0; index1 != dataset.rows; ++index1)
  {
    for (IndexType i = 0; i != knn + 1; ++i)
    {
      const IndexType index2 = indices[index1][i];
      if (index1 == index2) continue;
      if (index2 == -1) break;

      indices1.push_back(index1);
      indices2.push_back(index2);
    }
  }
}

template<int NumDimensions, typename ValueType, typename IndexType, typename PositionType>
void GenerateMinimumSpanningTreeWith(
  const std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType, ValueType)>& distanceFunc,
  const std::vector<ValueType>& positions,
  const std::vector<ValueType>& tangentLinesPoints1,
  const std::vector<ValueType>& tangentLinesPoints2,
  const std::vector<ValueType>& radiuses,
  std::vector<IndexType>& indices1,
  std::vector<IndexType>& indices2,
  int knn)
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

  GraphType origGraph(numberOfPoints);
  WeightMapType weightmap = get(edge_weight, origGraph);

  if (knn != -1)
  {
    GenerateKnnGraph(
      knn,
      positions,
      tangentLinesPoints1,
      tangentLinesPoints2,
      indices1,
      indices2,
      NumDimensions);

    GraphEdgeType e;

    for (size_t i = 0; i < indices1.size(); ++i)
    {
      const IndexType index1 = indices1[i];
      const IndexType index2 = indices2[i];

      tie(e, tuples::ignore) = add_edge(index1, index2, origGraph);

      const auto& point1 = positions_.row(index1);
      const auto& tangentLine1Point1 = tangentLinesPoints1_.row(index1);
      const auto& tangentLine1Point2 = tangentLinesPoints2_.row(index1);
      const ValueType radius1 = radiuses[index1];

      const auto& point2 = positions_.row(index2);
      const auto& tangentLine2Point1 = tangentLinesPoints1_.row(index2);
      const auto& tangentLine2Point2 = tangentLinesPoints2_.row(index2);
      const ValueType radius2 = radiuses[index2];

      const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2);
      weightmap[e] = distance;
    }

    indices1.clear();
    indices2.clear();
  }
  else
  {
    GraphEdgeType e;

    for (IndexType index1 = 0; index1 < numberOfPoints; ++index1)
    {
      for (IndexType index2 = index1 + 1; index2 < numberOfPoints; ++index2)
      {
        tie(e, tuples::ignore) = add_edge(index1, index2, origGraph);

        const auto& point1 = positions_.row(index1);
        const auto& tangentLine1Point1 = tangentLinesPoints1_.row(index1);
        const auto& tangentLine1Point2 = tangentLinesPoints2_.row(index1);
        const ValueType radius1 = radiuses[index1];

        const auto& point2 = positions_.row(index2);
        const auto& tangentLine2Point1 = tangentLinesPoints1_.row(index2);
        const auto& tangentLine2Point2 = tangentLinesPoints2_.row(index2);
        const ValueType radius2 = radiuses[index2];

        const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2);
        weightmap[e] = distance;
      }
    }
  }

  std::vector<GraphEdgeType> spanningTree;
  spanningTree.reserve(numberOfPoints - 1);

  kruskal_minimum_spanning_tree(origGraph, std::back_inserter(spanningTree));

  for (const GraphEdgeType& e : spanningTree)
  {
    const IndexType index1 = source(e, origGraph);
    const IndexType index2 = target(e, origGraph);

    indices1.push_back(index1);
    indices2.push_back(index2);
  }
}

enum class DistanceOptions
{
  Euclidean = 1,
  ArcLengthsSum = 2,
  ArcLengthsMin = 3
};

void DoGenerateMinimumSpanningTreeOnlyPositionsDataSet(const std::string& inputFileName, const std::string& outputFileName, int knn)
{
  const int NumDimensions = 3;

  typedef double ValueType;
  typedef int IndexType;

  typedef Eigen::Matrix<ValueType, 1, NumDimensions> PositionType;

  const std::string positionsDataSetName = "positions";

  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";

  BOOST_LOG_TRIVIAL(info) << "input filename = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\""; 
  BOOST_LOG_TRIVIAL(info) << "knn = " << knn;

  FileReader inputFileReader(inputFileName);

  std::vector<ValueType> positions;

  inputFileReader.Read(positionsDataSetName, positions);

  std::vector<IndexType> indices1;
  std::vector<IndexType> indices2;

  GenerateEuclideanMinimumSpanningTree<NumDimensions>(positions, indices1, indices2, knn);

  BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
  BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(positionsDataSetName, positions);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

void DoGenerateMinimumSpanningTree(const std::string& inputFileName, const std::string& outputFileName, bool noPositionsDataSet, DistanceOptions distanceOption, int knn)
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

  BOOST_LOG_TRIVIAL(info) << "input filename  = \"" << inputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "knn = " << knn;

  FileReader inputFileReader(inputFileName);

  std::vector<ValueType> measurements;
  std::vector<ValueType> positions;
  std::vector<ValueType> tangentLinesPoints1;
  std::vector<ValueType> tangentLinesPoints2;
  std::vector<ValueType> radiuses;
  std::vector<ValueType> objectnessMeasure;

  inputFileReader.Read(measurementsDataSetName, measurements);
  if (!noPositionsDataSet)
    inputFileReader.Read(positionsDataSetName, positions);
  inputFileReader.Read(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  inputFileReader.Read(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  inputFileReader.Read(radiusesDataSetName, radiuses);
  inputFileReader.Read(objectnessMeasureDataSetName, objectnessMeasure);

  const std::vector<ValueType>& points = noPositionsDataSet ? measurements : positions;

  std::vector<IndexType> indices1;
  std::vector<IndexType> indices2;

  switch (distanceOption)
  {
  case DistanceOptions::Euclidean:
    GenerateMinimumSpanningTreeWith<NumDimensions>(
      (DistanceFunctionType)ComputeEuclideanDistance<ValueType, PositionType>,
      points,
      tangentLinesPoints1,
      tangentLinesPoints2,
      radiuses,
      indices1,
      indices2,
      knn);
    break;

  case DistanceOptions::ArcLengthsSum:
    GenerateMinimumSpanningTreeWith<NumDimensions>(
      (DistanceFunctionType)ComputeArcLengthsSumDistance<ValueType, PositionType>,
      points,
      tangentLinesPoints1,
      tangentLinesPoints2,
      radiuses,
      indices1,
      indices2,
      knn);
    break;

  case DistanceOptions::ArcLengthsMin:
    GenerateMinimumSpanningTreeWith<NumDimensions>(
      (DistanceFunctionType)ComputeArcLengthsMinDistance<ValueType, PositionType>,
      points,
      tangentLinesPoints1,
      tangentLinesPoints2,
      radiuses,
      indices1,
      indices2,
      knn);
    break;

  default:
    throw std::invalid_argument("distanceOption is invalid");
  }

  BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
  BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(measurementsDataSetName, measurements);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
  outputFileWriter.Write(objectnessMeasureDataSetName, objectnessMeasure);
  if (!noPositionsDataSet)
    outputFileWriter.Write(positionsDataSetName, positions);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  std::string inputFileName;
  std::string outputFileName;

  bool noPositionsDataSet = false;
  bool onlyPositionsDataSet = false;

  int optionNum = (int)DistanceOptions::Euclidean;
  int knn = -1;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("inputFileName", po::value(&inputFileName)->required(), "the name of the input file")
    ("outputFileName", po::value(&outputFileName)->required(), "the name of the output file")
    ("noPositions", po::value(&noPositionsDataSet), "indicate that '/positions' dataset is not present in the input file")
    ("onlyPositions", po::value(&onlyPositionsDataSet), "indicate that only '/positions' dataset is present in the input file")
    ("optionNum", po::value(&optionNum), "the option number of distance function between two points")
    ("knn", po::value(&knn), "the number of nearest neighbors to consider (if not specified then use complete graph)");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    desc.print(std::cout);
    return EXIT_SUCCESS;
  }

  if (onlyPositionsDataSet)
  {
    try
    {
      DoGenerateMinimumSpanningTreeOnlyPositionsDataSet(inputFileName, outputFileName, knn);
      return EXIT_SUCCESS;
    }
    catch (std::exception& e)
    {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  try
  {
    DoGenerateMinimumSpanningTree(inputFileName, outputFileName, noPositionsDataSet, (DistanceOptions)optionNum, knn);
    return EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}