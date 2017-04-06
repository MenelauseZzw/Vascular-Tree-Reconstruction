#define BOOST_NO_CXX11_RVALUE_REFERENCES //http://linux.debian.devel.mentors.narkive.com/3WPG83eZ/use-of-deleted-function-boost-detail-stored-edge-property
#define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#include "FileReader.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/visitors.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <boost/unordered_map.hpp>
#include <Eigen/Dense>
#include <exception>
#include <flann/flann.hpp>
#include <forward_list>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#pragma region dp

double dp(
  int numNodes, int rootNode,
  const std::function<int(int)>& getNumChildren,
  const std::function<int(int, int)>& getNthChild,
  const std::function<int(int)>& getNumLabels,
  const std::function<int(int)>& getOptLabel,
  const std::function<void(int, int)>& setOptLabel,
  const std::function<double(int, int)>& getCost1,
  const std::function<double(int, int, int, int)>& getCost2)
{
  using std::for_each;
  using std::vector;

  constexpr int InvalidLabel = -1;

  vector<int> nodesInLevelOrder;

  nodesInLevelOrder.reserve(numNodes);
  nodesInLevelOrder.push_back(rootNode);

  for (int k = 0; k != numNodes; ++k)
  {
    const int curNode = nodesInLevelOrder[k];

    for (int i = 0; i != getNumChildren(curNode); ++i)
    {
      nodesInLevelOrder.push_back(getNthChild(curNode, i));
    }
  }

  vector<vector<double>> optLabelsCosts(numNodes);
  vector<vector<vector<int>>> childrenOptLabels(numNodes);

  for_each(nodesInLevelOrder.crbegin(), nodesInLevelOrder.crend(), [&](const int curNode)
  {
    optLabelsCosts[curNode].resize(getNumLabels(curNode));
    childrenOptLabels[curNode].resize(getNumLabels(curNode), vector<int>(getNumChildren(curNode)));

    for (int curLabel = 0; curLabel != getNumLabels(curNode); ++curLabel)
    {
      double curLabelCost = getCost1(curNode, curLabel);

      for (int i = 0; i != getNumChildren(curNode); ++i)
      {
        const int child = getNthChild(curNode, i);

        int optChildLabel = InvalidLabel;
        double optChildLabelCost = 0;

        for (int childLabel = 0; childLabel != getNumLabels(child); ++childLabel)
        {
          const double cost = getCost2(curNode, curLabel, i, childLabel) + optLabelsCosts[child][childLabel];

          if (optChildLabel == InvalidLabel || cost < optChildLabelCost)
          {
            optChildLabel = childLabel;
            optChildLabelCost = cost;
          }
        }

        childrenOptLabels[curNode][curLabel][i] = optChildLabel;
        curLabelCost += optChildLabelCost;
      }

      optLabelsCosts[curNode][curLabel] = curLabelCost;
    }
  });

  int optRootLabel = InvalidLabel;
  double optRootLabelCost = 0;

  for (int rootLabel = 0; rootLabel != getNumLabels(rootNode); ++rootLabel)
  {
    const double cost = optLabelsCosts[rootNode][rootLabel];

    if (optRootLabel == InvalidLabel || cost < optRootLabelCost)
    {
      optRootLabel = rootLabel;
      optRootLabelCost = cost;
    }
  }

  setOptLabel(rootNode, optRootLabel);

  for_each(nodesInLevelOrder.cbegin(), nodesInLevelOrder.cend(), [&](const int curNode)
  {
    for (int i = 0; i != getNumChildren(curNode); ++i)
    {
      setOptLabel(getNthChild(curNode, i), childrenOptLabels[curNode][getOptLabel(curNode)][i]);
    }
  });

  return optRootLabelCost;
}

#pragma endregion

template<typename PositionType>
struct SourceGraphNode
{
  PositionType Position;
};

template<typename ValueType>
struct SourceGraphEdge
{
  ValueType Radius;
};

template<typename PositionType>
struct TargetGraphNode
{
  PositionType Position;
};

template<int NumDimensions, typename ValueType, typename IndexType>
void ComputeSourceToTargetKnnLabels(
  const std::vector<ValueType>& sourcePositions,
  const std::vector<ValueType>& targetPositions,
  std::vector<std::vector<IndexType>>& sourceToTargetLabels,
  int numberOfLabels)
{
  using namespace flann;

  const Matrix<ValueType> sourcePoints(const_cast<ValueType*>(sourcePositions.data()), sourcePositions.size() / NumDimensions, NumDimensions);
  const Matrix<ValueType> targetPoints(const_cast<ValueType*>(targetPositions.data()), targetPositions.size() / NumDimensions, NumDimensions);

  BOOST_LOG_TRIVIAL(info) << "Computing labels lp(sourceGraph.V) = targetGraph.V using k-nn algorithm";

  Index<L2<ValueType>> index(targetPoints, KDTreeIndexParams(1));
  index.buildIndex();

  std::vector<std::vector<ValueType>> sourceToTargetDistances;
  index.knnSearch(sourcePoints, sourceToTargetLabels, sourceToTargetDistances, numberOfLabels, SearchParams(-1));
}

template<typename ValueType, typename GraphType>
ValueType ComputeEuclideanLengthOfGraph(const GraphType& g)
{
  using namespace std;
  using namespace boost;

  typedef typename graph_traits<GraphType>::edge_iterator GraphEdgeIteratorType;

  ValueType euclideanLength = 0;

  auto es = boost::edges(g);

  for (auto eit = es.first; eit != es.second; ++eit)
  {
    const auto index1 = source(*eit, g);
    const auto index2 = target(*eit, g);

    const auto& point1 = g[index1].Position;
    const auto& point2 = g[index2].Position;

    euclideanLength += (point1 - point2).norm();
  }

  return euclideanLength;
}

template<
  int NumDimensions,
  typename ValueType,
  typename IndexType,
  typename PositionType = Eigen::Matrix<ValueType, 1, NumDimensions>,
  typename SourceGraphNodeType = SourceGraphNode<PositionType>,
  typename SourceGraphEdgeType = SourceGraphEdge<ValueType>,
  typename SourceGraphType = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, SourceGraphNodeType, SourceGraphEdgeType>>
  SourceGraphType GenerateSourceGraph(
    const std::vector<ValueType>& positions,
    const std::vector<IndexType>& indices1,
    const std::vector<IndexType>& indices2,
    const std::vector<ValueType>& radiuses,
    IndexType graphRoot)
{
  using namespace std;
  using namespace boost;
  using namespace Eigen;

  typedef adjacency_list<vecS, vecS, undirectedS> UndirectedGraphType;
  typedef typename graph_traits<SourceGraphType>::edge_descriptor SourceGraphEdgeType;

  const size_t numSourceNodes = radiuses.size();
  const size_t numSourceEdges = indices1.size();

  UndirectedGraphType originalGraph(numSourceNodes);

  for (size_t i = 0; i < numSourceEdges; ++i)
  {
    const IndexType sourceIndex1 = indices1[i];
    const IndexType sourceIndex2 = indices2[i];

    add_edge(sourceIndex1, sourceIndex2, originalGraph);
  }

  vector<IndexType> predecessors(numSourceNodes);
  depth_first_search(originalGraph, visitor(make_dfs_visitor(record_predecessors(&predecessors[0], on_tree_edge()))).root_vertex(graphRoot));

  SourceGraphType sourceGraph(numSourceNodes);

  const ValueType* pSourcePositions = &positions[0];

  for (IndexType sourceIndex = 0; sourceIndex < numSourceNodes; ++sourceIndex, pSourcePositions += NumDimensions)
  {
    sourceGraph[sourceIndex].Position = Map<const PositionType>(pSourcePositions, NumDimensions);
  }

  for (size_t i = 0; i < numSourceEdges; ++i)
  {
    const IndexType sourceIndex1 = indices1[i];
    const IndexType sourceIndex2 = indices2[i];
    const ValueType radius = radiuses[i];

    SourceGraphEdgeType e;

    if (sourceIndex1 == predecessors[sourceIndex2])
    {
      tie(e, tuples::ignore) = add_edge(sourceIndex1, sourceIndex2, sourceGraph);
    }
    else
    {
      tie(e, tuples::ignore) = add_edge(sourceIndex2, sourceIndex1, sourceGraph);
    }

    sourceGraph[e].Radius = radius;
  }

  BOOST_LOG_TRIVIAL(info) << "sourceGraph.|V|    = " << num_vertices(sourceGraph);
  BOOST_LOG_TRIVIAL(info) << "sourceGraph.|E|    = " << num_edges(sourceGraph);

  return sourceGraph;
}

template<size_t NumDimensions,
  typename ValueType,
  typename IndexType,
  typename PositionType = Eigen::Matrix<ValueType, 1U, NumDimensions>,
  typename TargetGraphNodeType = TargetGraphNode<PositionType>,
  typename TargetGraphType = boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, TargetGraphNodeType>>
  TargetGraphType GenerateTargetGraph(
    const std::vector<ValueType>& positions,
    const std::vector<IndexType>& indices1,
    const std::vector<IndexType>& indices2)
{
  using namespace std;
  using namespace boost;
  using namespace Eigen;

  const size_t numTargetNodes = positions.size() / NumDimensions;
  const size_t numTargetEdges = indices1.size();

  TargetGraphType targetGraph(numTargetNodes);

  const ValueType* pTargetPositions = &positions[0];

  for (IndexType targetIndex = 0; targetIndex < numTargetNodes; ++targetIndex, pTargetPositions += NumDimensions)
  {
    targetGraph[targetIndex].Position = Map<const PositionType>(pTargetPositions, NumDimensions);
  }

  for (size_t i = 0; i < indices1.size(); ++i)
  {
    const IndexType targetIndex1 = indices1[i];
    const IndexType targetIndex2 = indices2[i];

    add_edge(targetIndex1, targetIndex2, targetGraph);
  }

  BOOST_LOG_TRIVIAL(info) << "targetGraph.|V|    = " << num_vertices(targetGraph);
  BOOST_LOG_TRIVIAL(info) << "targetGraph.|E|    = " << num_edges(targetGraph);

  return targetGraph;
}

template<typename ValueType, typename IndexType, typename GraphType, typename PositionType>
void GeneratePolyLinePoints(
  const GraphType& g,
  IndexType index1,
  IndexType index2,
  ValueType eps,
  std::vector<PositionType>& polyLinePoints)
{
  using namespace std;
  using namespace boost;

  const auto& point1 = g[index1].Position;
  const auto& point2 = g[index2].Position;

  const ValueType distance = (point1 - point2).norm();

  if (distance < eps)
  {
    polyLinePoints.push_back(point2);
  }
  else
  {
    const IndexType numIntervals = (IndexType)(distance / eps) + 1;

    for (IndexType k = 0; k < numIntervals; ++k)
    {
      const ValueType alpha = (ValueType)k / numIntervals;
      const auto x = (1 - alpha) * point2 + alpha * point1;
      polyLinePoints.push_back(x);
    }
  }

  polyLinePoints.push_back(point1);
}

template<typename ValueType, typename IndexType, typename GraphType, typename PositionType>
void GeneratePolyLinePoints(const GraphType& g, IndexType nodeIndex1, IndexType nodeIndex2, const std::vector<IndexType>& predecessors, ValueType eps, std::vector<PositionType>& polyLine)
{
  for (IndexType index2 = nodeIndex2; index2 != nodeIndex1; index2 = predecessors[index2])
  {
    const IndexType index1 = predecessors[index2];
    GeneratePolyLinePoints(g, index1, index2, eps, polyLine);
    polyLine.pop_back();
  }

  const auto& node1 = g[nodeIndex1].Position;
  polyLine.push_back(node1);
}

template<typename ValueType>
ValueType DiscreteFrechetDistance(size_t p, size_t q, const std::function<ValueType(size_t, size_t)>& distanceFunc)
{
  using namespace std;

  vector<vector<ValueType>> ca(p, vector<ValueType>(q));

  ca[0][0] = distanceFunc(0, 0);

  for (size_t j = 1; j != q; ++j)
  {
    ca[0][j] = max(ca[0][j - 1], distanceFunc(0, j));
  }

  for (size_t i = 1; i != p; ++i)
  {
    ca[i][0] = max(ca[i - 1][0], distanceFunc(i, 0));
  }

  for (size_t i = 1; i != p; ++i)
  {
    for (size_t j = 1; j != q; ++j)
    {
      ca[i][j] = max(min({ ca[i - 1][j], ca[i - 1][j - 1], ca[i][j - 1] }), distanceFunc(i, j));
    }
  }

  return ca[p - 1][q - 1];
}

template<int NumDimensions, typename ValueType, typename IndexType, typename SourceGraphType, typename TargetGraphType>
void GenerateSourceToTargetDistances(
  const SourceGraphType& sourceGraph,
  const TargetGraphType& targetGraph,
  const std::vector<std::vector<IndexType>>& sourceToTargetLabels,
  std::vector<std::vector<std::vector<std::vector<ValueType>>>>& sourceToTargetDistances,
  double voxelPhysicalSize)
{
  using namespace std;
  using namespace boost;
  using namespace Eigen;

  typedef Matrix<ValueType, 1, NumDimensions> PositionType;

  const ValueType eps = voxelPhysicalSize / 10;//frechetDistance < discreteFrechetDistance < frechetDistance + eps

  const size_t numSourceNodes = num_vertices(sourceGraph);
  const size_t numTargetNodes = num_vertices(targetGraph);
  sourceToTargetDistances.resize(numSourceNodes);

  BOOST_LOG_TRIVIAL(info) << "Computing Frechet distance between (u,v) in sourceGraph.E and (lp(u),lp(v)) in targetGraph.E";

  for (IndexType sourceIndex1 = 0; sourceIndex1 < numSourceNodes; ++sourceIndex1)
  {
    const PositionType& sourcePoint1 = sourceGraph[sourceIndex1].Position;

    const auto& sourceToTargetLabels1 = sourceToTargetLabels[sourceIndex1];
    const IndexType numSourceLabels1 = sourceToTargetLabels1.size() + 1;//plus an ``outlier`` label

    sourceToTargetDistances[sourceIndex1].resize(numSourceLabels1);

    const IndexType numChildren = out_degree(sourceIndex1, sourceGraph);

    typename graph_traits<SourceGraphType>::out_edge_iterator childIterator, childIteratorEnd;
    tie(childIterator, childIteratorEnd) = out_edges(sourceIndex1, sourceGraph);

    for (IndexType childIndex = 0; childIndex < numChildren; ++childIndex, ++childIterator)
    {
      const auto e = *childIterator;

      const IndexType sourceIndex2 = target(e, sourceGraph);
      const PositionType& sourcePoint2 = sourceGraph[sourceIndex2].Position;

      const ValueType sourceRadius = sourceGraph[e].Radius;

      const auto& sourceToTargetLabels2 = sourceToTargetLabels[sourceIndex2];
      const IndexType numSourceLabels2 = sourceToTargetLabels2.size() + 1;//plus an ``outlier`` label

      const ValueType distanceBetweenSourcePoints = (sourcePoint1 - sourcePoint2).norm();

      for (IndexType sourceLabel1 = 0; sourceLabel1 < numSourceLabels1; ++sourceLabel1)
      {
        sourceToTargetDistances[sourceIndex1][sourceLabel1].resize(numChildren);

        if (sourceLabel1 != numSourceLabels1 - 1)
        {
          const IndexType targetIndex1 = sourceToTargetLabels1[sourceLabel1];

          vector<IndexType> targetDistances(numTargetNodes);
          vector<IndexType> targetPredecessors(numTargetNodes);

          const auto targetBfsVisitor = make_bfs_visitor(make_pair(
            record_predecessors(&targetPredecessors[0], on_tree_edge()),
            record_distances(&targetDistances[0], on_tree_edge())));

          breadth_first_search(targetGraph, targetIndex1, visitor(targetBfsVisitor));

          sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex].resize(numSourceLabels2);

          for (IndexType sourceLabel2 = 0; sourceLabel2 < sourceToTargetLabels2.size(); ++sourceLabel2)
          {
            const IndexType targetIndex2 = sourceToTargetLabels2[sourceLabel2];

            auto& sourceToTargetDistance = sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex][sourceLabel2];

            if (targetIndex1 == targetIndex2)//forbidden situation
            {
              sourceToTargetDistance = numeric_limits<ValueType>::infinity();
            }
            else if (targetDistances[targetIndex2] > 0)//there is a path between nodes ``targetIndex1`` and ``targetIndex2``
            {
              vector<PositionType> sourcePolyLinePoints;
              GeneratePolyLinePoints(sourceGraph, sourceIndex1, sourceIndex2, eps, sourcePolyLinePoints);

              vector<PositionType> targetPolyLinePoints;
              GeneratePolyLinePoints(targetGraph, targetIndex1, targetIndex2, targetPredecessors, eps, targetPolyLinePoints);

              const ValueType frechetDistance = DiscreteFrechetDistance<ValueType>(sourcePolyLinePoints.size(), targetPolyLinePoints.size(), [&](size_t sourceIndex, size_t targetIndex)
              {
                return (sourcePolyLinePoints[sourceIndex] - targetPolyLinePoints[targetIndex]).norm();
              });

              sourceToTargetDistance = frechetDistance / sourceRadius;
            }
            else//there is no a path between nodes ``targetIndex1`` and ``targetIndex2``
            {
              sourceToTargetDistance = -distanceBetweenSourcePoints;
            }
          }

          sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex][numSourceLabels2 - 1] = -distanceBetweenSourcePoints;
        }
        else//sourceLabel1 = numSourceLabels1 - 1
        {
          sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex].assign(numSourceLabels2, -distanceBetweenSourcePoints);
        }
      }
    }
  }
}

template<typename ValueType, typename IndexType, typename SourceGraphType, typename TargetGraphType>
ValueType ComputeSourceToTargetOptLabels(
  const SourceGraphType& sourceGraph,
  const TargetGraphType& targetGraph,
  const std::vector<std::vector<IndexType>>& sourceToTargetLabels,
  const std::vector<std::vector<std::vector<std::vector<ValueType>>>>& sourceToTargetDistances,
  IndexType sourceGraphRoot,
  ValueType parValue,
  std::vector<IndexType>& sourceToTargetOptLabels,
  std::vector<ValueType>& sourceToTargetGraphsPositiveDistances)
{
  using namespace std;
  using namespace boost;

  typedef typename graph_traits<SourceGraphType>::adjacency_iterator SourceGraphAdjacencyIteratorType;
  typedef typename graph_traits<SourceGraphType>::edge_descriptor SourceGraphEdgeType;

  const size_t numSourceNodes = num_vertices(sourceGraph);
  const size_t numTargetNodes = num_vertices(targetGraph);

  const std::function<IndexType(IndexType)> getNumChildren = [&sourceGraph](IndexType sourceIndex)
  {
    return out_degree(sourceIndex, sourceGraph);
  };

  const std::function<IndexType(IndexType, IndexType)> getNthChild = [&sourceGraph](IndexType curNode, IndexType childIndex)
  {
    SourceGraphAdjacencyIteratorType adjacencyIterator;
    tie(adjacencyIterator, tuples::ignore) = adjacent_vertices(curNode, sourceGraph);
    return *(adjacencyIterator + childIndex);
  };

  const std::function<IndexType(IndexType)> getNumLabels = [&sourceToTargetLabels](IndexType curNode)
  {
    return sourceToTargetLabels[curNode].size() + 1;
  };

  vector<IndexType> optLabels(numSourceNodes);

  const std::function<IndexType(IndexType)> getOptLabel = [&optLabels](IndexType curNode)
  {
    return optLabels[curNode];
  };

  const std::function<void(IndexType, IndexType)> setOptlabel = [&optLabels](IndexType curNode, IndexType label)
  {
    optLabels[curNode] = label;
  };

  const std::function<ValueType(IndexType, IndexType)> func1 = [getNumLabels](IndexType curNode, IndexType curLabel)
  {
    return curLabel != getNumLabels(curNode) ? 0 : -numeric_limits<ValueType>::min();
  };

  const std::function<ValueType(IndexType, IndexType, IndexType, IndexType)> func2 = [getNthChild, parValue, &sourceToTargetDistances](IndexType curNode, IndexType curLabel, IndexType child, IndexType childLabel)
  {
    const IndexType childNode = getNthChild(curNode, child);
    const ValueType distance = sourceToTargetDistances[curNode][curLabel][child][childLabel];

    return (distance < 0) ? (-distance * parValue) : distance;
  };

  const ValueType costDP = dp(
    numSourceNodes, sourceGraphRoot, getNumChildren, getNthChild, getNumLabels, getOptLabel, setOptlabel, func1, func2);

  sourceToTargetOptLabels.resize(numSourceNodes);

  for (IndexType curNode = 0; curNode < numSourceNodes; ++curNode)
  {
    const auto& curNodeLabels = sourceToTargetLabels[curNode];
    IndexType& sourceToTargetOptLabel = sourceToTargetOptLabels[curNode];

    if (optLabels[curNode] != curNodeLabels.size())
    {
      sourceToTargetOptLabel = curNodeLabels[optLabels[curNode]];
    }
    else
    {
      sourceToTargetOptLabel = -1;
    }
  }

  sourceToTargetGraphsPositiveDistances.clear();

  for (IndexType curNode = 0; curNode < numSourceNodes; ++curNode)
  {
    const IndexType curLabel = optLabels[curNode];

    for (IndexType child = 0; child < getNumChildren(curNode); ++child)
    {
      const IndexType childNode = getNthChild(curNode, child);
      const IndexType childLabel = optLabels[childNode];

      SourceGraphEdgeType e;
      tie(e, tuples::ignore) = edge(curNode, childNode, sourceGraph);

      const ValueType sourceToTargetGraphsDistance = sourceToTargetDistances[curNode][curLabel][child][childLabel];

      if (sourceToTargetGraphsDistance > 0)
      {
        sourceToTargetGraphsPositiveDistances.push_back(sourceToTargetGraphsDistance);
      }
    }
  }

  return costDP;
}

template<typename SourceGraphType, typename IndexType,
  typename SourceGraphEdgeType = typename boost::graph_traits<SourceGraphType>::edge_descriptor,
  typename SourceGraphEdgePredicateType = std::function<bool(const SourceGraphEdgeType&)>,
  typename SourcePrimeGraphType = boost::filtered_graph < SourceGraphType, SourceGraphEdgePredicateType >>
  SourcePrimeGraphType GenerateSourcePrimeGraph(const SourceGraphType& sourceGraph, const std::vector<IndexType>& sourceToTargetOptLabels)
{
  using namespace std;
  using namespace boost;

  const SourceGraphEdgePredicateType sourceGraphEdgePredicate = [&sourceGraph, &sourceToTargetOptLabels](const SourceGraphEdgeType& e)
  {
    const IndexType sourceIndex1 = source(e, sourceGraph);
    const IndexType sourceIndex2 = target(e, sourceGraph);

    const IndexType targetIndex1 = sourceToTargetOptLabels[sourceIndex1];
    const IndexType targetIndex2 = sourceToTargetOptLabels[sourceIndex2];

    return (targetIndex1 != -1) && (targetIndex2 != -1);
  };

  return SourcePrimeGraphType(sourceGraph, sourceGraphEdgePredicate);
}

template<typename SourceGraphType, typename TargetGraphType, typename IndexType,
  typename TargetGraphEdgeType = typename boost::graph_traits<TargetGraphType>::edge_descriptor,
  typename TargetGraphEdgePredicateType = std::function<bool(const TargetGraphEdgeType&)>,
  typename TargetPrimeGraphType = boost::filtered_graph < TargetGraphType, TargetGraphEdgePredicateType >>
  TargetPrimeGraphType GenerateTargetPrimeGraph(const SourceGraphType& sourceGraph, const TargetGraphType& targetGraph, const std::vector<IndexType>& sourceToTargetOptLabels)
{
  using namespace std;
  using namespace boost;

  typedef typename graph_traits<SourceGraphType>::edge_iterator SourceGraphEdgeIteratorType;
  typedef typename graph_traits<TargetGraphType>::edge_iterator TargetGraphEdgeIteratorType;

  const size_t numSourceNodes = num_vertices(sourceGraph);
  const size_t numTargetNodes = num_vertices(targetGraph);

  SourceGraphEdgeIteratorType sourceEdgeIterator, sourceEdgeIteratorEnd;
  tie(sourceEdgeIterator, sourceEdgeIteratorEnd) = edges(sourceGraph);

  vector<IndexType> predecessors(numTargetNodes);

  boost::unordered_set<TargetGraphEdgeType> targetPrimeGraphEdges;

  for (; sourceEdgeIterator != sourceEdgeIteratorEnd; ++sourceEdgeIterator)
  {
    const auto e = *sourceEdgeIterator;

    const IndexType sourceIndex1 = source(e, sourceGraph);
    const IndexType sourceIndex2 = target(e, sourceGraph);

    const IndexType targetIndex1 = sourceToTargetOptLabels[sourceIndex1];
    const IndexType targetIndex2 = sourceToTargetOptLabels[sourceIndex2];

    if (targetIndex1 != -1 && targetIndex2 != -1)
    {
      breadth_first_search(targetGraph, targetIndex1, visitor(make_bfs_visitor(record_predecessors(&predecessors[0], on_tree_edge()))));

      for (IndexType index2 = targetIndex2; index2 != targetIndex1; index2 = predecessors[index2])
      {
        const IndexType index1 = predecessors[index2];

        TargetGraphEdgeType e;
        tie(e, tuples::ignore) = edge(index1, index2, targetGraph);
        targetPrimeGraphEdges.insert(e);
      }
    }
  }

  const TargetGraphEdgePredicateType targetGraphEdgePredicate = [targetPrimeGraphEdges](const TargetGraphEdgeType& e)
  {
    return targetPrimeGraphEdges.count(e) != 0;
  };

  return TargetPrimeGraphType(targetGraph, targetGraphEdgePredicate);
}

template<typename SourceGraphType, typename ValueType, typename IndexType>
void GenerateSourceGraphDataSet(const SourceGraphType& sourceGraph, std::vector<IndexType>& sourceIndices1, std::vector<IndexType>& sourceIndices2, std::vector<ValueType>& sourceRadiusesPrime)
{
  using namespace std;
  using namespace boost;

  typedef typename graph_traits<SourceGraphType>::edge_iterator SourceGraphEdgeIteratorType;

  SourceGraphEdgeIteratorType sourceEdgeIterator, sourceEdgeIteratorEnd;
  tie(sourceEdgeIterator, sourceEdgeIteratorEnd) = edges(sourceGraph);

  for (; sourceEdgeIterator != sourceEdgeIteratorEnd; ++sourceEdgeIterator)
  {
    const auto e = *sourceEdgeIterator;

    const IndexType sourceIndex1 = source(e, sourceGraph);
    const IndexType sourceIndex2 = target(e, sourceGraph);
    const ValueType sourceRadiusPrime = sourceGraph[e].Radius;

    sourceIndices1.push_back(sourceIndex1);
    sourceIndices2.push_back(sourceIndex2);
    sourceRadiusesPrime.push_back(sourceRadiusPrime);
  }
}

template<typename TargetGraphType, typename IndexType>
void GenerateTargetGraphDataSet(const TargetGraphType& targetGraph, std::vector<IndexType>& targetIndices1, std::vector<IndexType>& targetIndices2)
{
  using namespace std;
  using namespace boost;

  typedef typename graph_traits<TargetGraphType>::edge_iterator TargetGraphEdgeIteratorType;

  TargetGraphEdgeIteratorType targetEdgeIterator, targetEdgeIteratorEnd;
  tie(targetEdgeIterator, targetEdgeIteratorEnd) = edges(targetGraph);

  for (; targetEdgeIterator != targetEdgeIteratorEnd; ++targetEdgeIterator)
  {
    const auto e = *targetEdgeIterator;

    const IndexType targetIndex1 = source(e, targetGraph);
    const IndexType targetIndex2 = target(e, targetGraph);

    targetIndices1.push_back(targetIndex1);
    targetIndices2.push_back(targetIndex2);
  }
}

void DoFrechetDistance(std::string const& sourceFileName, std::string const& targetFileName, std::string const& outputFileName, int numberOfNearestNeighbors, int sourceGraphRoot, double voxelPhysicalSize, int numberOfParValues, double parValueMinimum, double parValueMaximum)
{
  constexpr int NumDimensions = 3;

  typedef double ValueType;
  typedef int IndexType;

  BOOST_LOG_TRIVIAL(info) << "source filename = \"" << sourceFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "target filename = \"" << targetFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "output filename = \"" << outputFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "number of nearest neighbors (k) = " << numberOfNearestNeighbors;
  BOOST_LOG_TRIVIAL(info) << "voxel physical size = " << voxelPhysicalSize;

  const std::string positionsDataSetName = "positions";
  const std::string measurementsDataSetName = "measurements";
  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";

  FileReader sourceFileReader(sourceFileName);

  std::vector<ValueType> sourcePositions;
  std::vector<IndexType> sourceIndices1;
  std::vector<IndexType> sourceIndices2;
  std::vector<ValueType> sourceRadiuses;

  sourceFileReader.Read(positionsDataSetName, sourcePositions);
  sourceFileReader.Read(indices1DataSetName, sourceIndices1);
  sourceFileReader.Read(indices2DataSetName, sourceIndices2);
  sourceFileReader.Read(radiusesDataSetName, sourceRadiuses);

  FileReader targetFileReader(targetFileName);

  std::vector<ValueType> targetPositions;
  std::vector<IndexType> targetIndices1;
  std::vector<IndexType> targetIndices2;

  targetFileReader.Read(positionsDataSetName, targetPositions);
  targetFileReader.Read(indices1DataSetName, targetIndices1);
  targetFileReader.Read(indices2DataSetName, targetIndices2);

  std::fstream outputFile(outputFileName, std::ios::out);

  const auto sourceGraph = GenerateSourceGraph<NumDimensions>(sourcePositions, sourceIndices1, sourceIndices2, sourceRadiuses, sourceGraphRoot);
  const ValueType sourceGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(sourceGraph);
  BOOST_LOG_TRIVIAL(info) << "sourceGraph.length = " << sourceGraphLength;

  const auto targetGraph = GenerateTargetGraph<NumDimensions>(targetPositions, targetIndices1, targetIndices2);
  const ValueType targetGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(targetGraph);
  BOOST_LOG_TRIVIAL(info) << "targetGraph.length = " << targetGraphLength;

  std::vector<std::vector<IndexType>> knnLabels;
  ComputeSourceToTargetKnnLabels<NumDimensions>(sourcePositions, targetPositions, knnLabels, numberOfNearestNeighbors);

  std::vector<std::vector<std::vector<std::vector<ValueType>>>> sourceToTargetDistances;
  GenerateSourceToTargetDistances<NumDimensions>(sourceGraph, targetGraph, knnLabels, sourceToTargetDistances, voxelPhysicalSize);

  std::vector<ValueType> parValues(numberOfParValues);
  std::vector<ValueType> sourceToTargetGraphsMaxDistances(numberOfParValues);
  std::vector<ValueType> sourceToTargetGraphsSumDistances(numberOfParValues);
  std::vector<ValueType> sourceToTargetGraphsAveDistances(numberOfParValues);
  std::vector<ValueType> costFunctionValues(numberOfParValues);
  std::vector<ValueType> sourceGraphsLengthsRatios(numberOfParValues);
  std::vector<ValueType> targetGraphsLengthsRatios(numberOfParValues);

  const ValueType parValueStep = (parValueMaximum - parValueMinimum) / (numberOfParValues - 1);

#pragma omp parallel for
  for (int k = 0; k < numberOfParValues; ++k)
  {
    const ValueType kthParValues = parValueMinimum + k * parValueStep;

    parValues[k] = kthParValues;

    std::vector<ValueType> sourceToTargetGraphsPositiveDistances;

    std::vector<IndexType> sourceToTargetOptLabels;
    costFunctionValues[k] = ComputeSourceToTargetOptLabels(
      sourceGraph,
      targetGraph,
      knnLabels,
      sourceToTargetDistances,
      sourceGraphRoot,
      kthParValues,
      sourceToTargetOptLabels,
      sourceToTargetGraphsPositiveDistances);

    if (sourceToTargetGraphsPositiveDistances.empty())
    {
      sourceToTargetGraphsMaxDistances[k] = 0;
      sourceToTargetGraphsSumDistances[k] = 0;
      sourceToTargetGraphsAveDistances[k] = 0;

      sourceGraphsLengthsRatios[k] = 0;
      targetGraphsLengthsRatios[k] = 0;
    }
    else
    {
      sourceToTargetGraphsMaxDistances[k] = *std::max_element(sourceToTargetGraphsPositiveDistances.cbegin(), sourceToTargetGraphsPositiveDistances.cend());
      sourceToTargetGraphsSumDistances[k] = std::accumulate(sourceToTargetGraphsPositiveDistances.cbegin(), sourceToTargetGraphsPositiveDistances.cend(), ValueType(0));
      sourceToTargetGraphsAveDistances[k] = sourceToTargetGraphsSumDistances[k] / sourceToTargetGraphsPositiveDistances.size();

      const auto sourcePrimeGraph = GenerateSourcePrimeGraph(sourceGraph, sourceToTargetOptLabels);
      const ValueType sourcePrimeGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(sourcePrimeGraph);
      sourceGraphsLengthsRatios[k] = sourcePrimeGraphLength / sourceGraphLength;

      const auto targetPrimeGraph = GenerateTargetPrimeGraph(sourceGraph, targetGraph, sourceToTargetOptLabels);
      const ValueType targetPrimeGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(targetPrimeGraph);
      targetGraphsLengthsRatios[k] = targetPrimeGraphLength / targetGraphLength;
    }
  }

  outputFile << "ParValue,SourceToTargetGraphsAveDistance,SourceToTargetGraphsMaxDistance,SourceToTargetGraphsSumDistance,SourceGraphsLengthsRatio,TargetGraphsLengthsRatio,CostFunctionValue" << std::endl;

  for (int k = 0; k < numberOfParValues; ++k)
  {
    const ValueType kthParValues = parValues[k];
    const ValueType sourceToTargetGraphsMaxDistance = sourceToTargetGraphsMaxDistances[k];
    const ValueType sourceToTargetGraphsSumDistance = sourceToTargetGraphsSumDistances[k];
    const ValueType sourceToTargetGraphsAveDistance = sourceToTargetGraphsAveDistances[k];
    const ValueType sourceGraphsLengthsRatio = sourceGraphsLengthsRatios[k];
    const ValueType targetGraphsLengthsRatio = targetGraphsLengthsRatios[k];
    const ValueType costFunctionValue = costFunctionValues[k];

    outputFile << std::fixed << std::setprecision(3)
      << kthParValues << ',' << std::setprecision(5)
      << sourceToTargetGraphsAveDistance << ','
      << sourceToTargetGraphsMaxDistance << ','
      << sourceToTargetGraphsSumDistance << ','
      << sourceGraphsLengthsRatio << ','
      << targetGraphsLengthsRatio << ','
      << costFunctionValue << std::endl;
  }
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  std::string sourceFileName;
  std::string targetFileName;
  std::string outputFileName;
  double voxelPhysicalSize;

  int numberOfParValues = 100;
  double parValueMaximum = 1;
  double parValueMinimum = 0;

  int numberOfNearestNeighbors = 17;
  int sourceGraphRoot = 0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("sourceFileName", po::value(&sourceFileName)->required(), "the name of source file")
    ("targetFileName", po::value(&targetFileName)->required(), "the name of target file")
    ("outputFileName", po::value(&outputFileName)->required(), "the name of output file")
    ("voxelPhysicalSize", po::value(&voxelPhysicalSize)->required(), "the physical size of a voxel")
    ("numberOfParValues", po::value(&numberOfParValues), "the number of parameter values")
    ("parValueMaximum", po::value(&parValueMaximum), "the maximum value that the parameter should have")
    ("parValueMinimum", po::value(&parValueMinimum), "the minimum value that the parameter should have")
    ("numberOfNearestNeighbors", po::value(&numberOfNearestNeighbors), "number of nearest neighbors")
    ("sourceGraphRoot", po::value(&sourceGraphRoot), "root node index");

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
    DoFrechetDistance(sourceFileName, targetFileName, outputFileName, numberOfNearestNeighbors, sourceGraphRoot, voxelPhysicalSize, numberOfParValues, parValueMinimum, parValueMaximum);
    return EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
