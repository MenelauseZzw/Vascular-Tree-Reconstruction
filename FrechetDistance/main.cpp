#define BOOST_NO_CXX11_RVALUE_REFERENCES //http://linux.debian.devel.mentors.narkive.com/3WPG83eZ/use-of-deleted-function-boost-detail-stored-edge-property
#define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#include "FileReader.hpp"
#include "FileWriter.hpp"
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
#include <cassert>
#include <Eigen/Dense>
#include <flann/flann.hpp>
#include <forward_list>
#include <iostream>
#include <limits>
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
void GenerateSourceToTargetKnnLabels(
  const std::vector<ValueType>& sourcePositions, 
  const std::vector<ValueType>& targetPositions, 
  std::vector<std::vector<IndexType>>& sourceToTargetLabels, 
  int numberOfLabels)
{
  using namespace flann;

  const Matrix<ValueType> sourceNodePositions(const_cast<ValueType*>(sourcePositions.data()), sourcePositions.size() / NumDimensions, NumDimensions);
  const Matrix<ValueType> targetNodePositions(const_cast<ValueType*>(targetPositions.data()), targetPositions.size() / NumDimensions, NumDimensions);

  Index<L2<ValueType>> index(targetNodePositions, KDTreeIndexParams(1));
  index.buildIndex();

  std::vector<std::vector<ValueType>> sourceToTargetDistances;
  index.knnSearch(sourceNodePositions, sourceToTargetLabels, sourceToTargetDistances, numberOfLabels, SearchParams(-1));
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
  BOOST_LOG_TRIVIAL(info) << "sourceGraph.length = " << ComputeEuclideanLengthOfGraph<ValueType>(sourceGraph);

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

  BOOST_LOG_TRIVIAL(info) << "targetGraph.|V| = " << num_vertices(targetGraph);
  BOOST_LOG_TRIVIAL(info) << "targetGraph.|E| = " << num_edges(targetGraph);
  BOOST_LOG_TRIVIAL(info) << "targetGraph.|E| = " << ComputeEuclideanLengthOfGraph<ValueType>(targetGraph);

  return targetGraph;
}

template<typename ValueType, typename IndexType, typename PositionMap, typename PositionType>
void GeneratePolyLinePoints(const PositionMap& positionMap, IndexType nodeIndex1, IndexType nodeIndex2, ValueType eps, std::vector<PositionType>& polyLine)
{
  using namespace std;
  using namespace boost;

  const auto& node1 = positionMap[nodeIndex1].Position;
  const auto& node2 = positionMap[nodeIndex2].Position;

  const ValueType distance = (node1 - node2).norm();

  if (distance < eps)
  {
    polyLine.push_back(node2);
  }
  else
  {
    const IndexType numIntervals = (IndexType)(distance / eps) + 1;

    for (IndexType k = 0; k < numIntervals; ++k)
    {
      const ValueType alpha = (ValueType)k / numIntervals;
      const auto x = (1 - alpha) * node2 + alpha * node1;
      polyLine.push_back(x);
    }
  }

  polyLine.push_back(node1);
}

template<typename ValueType, typename IndexType, typename PositionMap, typename PositionType>
void GeneratePolyLinePoints(const PositionMap& positionMap, IndexType nodeIndex1, IndexType nodeIndex2, const std::vector<IndexType>& predecessors, ValueType eps, std::vector<PositionType>& polyLine)
{
  for (IndexType index2 = nodeIndex2; index2 != nodeIndex1; index2 = predecessors[index2])
  {
    const IndexType index1 = predecessors[index2];
    GeneratePolyLinePoints(positionMap, index1, index2, eps, polyLine);
    polyLine.pop_back();
  }

  const auto& node1 = positionMap[nodeIndex1].Position;
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

template<size_t NumDimensions, typename ValueType, typename IndexType, typename SourceGraphType, typename TargetGraphType>
void GenerateSourceToTargetDistances(const SourceGraphType& sourceGraph, const TargetGraphType& targetGraph, const std::vector<std::vector<IndexType>>& sourceToTargetLabels, std::vector<std::vector<std::vector<std::vector<ValueType>>>>& sourceToTargetDistances)
{
  using namespace std;
  using namespace boost;
  using namespace Eigen;

  typedef Matrix<ValueType, 1U, NumDimensions> PositionType;

  const ValueType voxelPhysicalSize = 0.046;
  const ValueType eps = voxelPhysicalSize / 10;//frechetDistance < discreteFrechetDistance < frechetDistance + eps

  const size_t numSourceNodes = num_vertices(sourceGraph);
  const size_t numTargetNodes = num_vertices(targetGraph);
  sourceToTargetDistances.resize(numSourceNodes);

  for (IndexType sourceIndex1 = 0; sourceIndex1 < numSourceNodes; ++sourceIndex1)
  {
    const PositionType& sourceNode1 = sourceGraph[sourceIndex1].Position;

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
      const PositionType& sourceNode2 = sourceGraph[sourceIndex2].Position;

      const ValueType sourceRadius = sourceGraph[e].Radius;

      const auto& sourceToTargetLabels2 = sourceToTargetLabels[sourceIndex2];
      const IndexType numSourceLabels2 = sourceToTargetLabels2.size() + 1;//plus an ``outlier`` label

      const ValueType sourceDistance = (sourceNode1 - sourceNode2).norm();

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
              sourceToTargetDistance = -sourceDistance;
            }
          }

          sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex][numSourceLabels2 - 1] = -sourceDistance;
        }
        else//sourceLabel1 = numSourceLabels1 - 1
        {
          sourceToTargetDistances[sourceIndex1][sourceLabel1][childIndex].assign(numSourceLabels2, -sourceDistance);
        }
      }
    }
  }
}

template<typename ValueType, typename IndexType, typename SourceGraphType, typename TargetGraphType>
ValueType GenerateSourceToTargetOptLabels(
  const SourceGraphType& sourceGraph, const TargetGraphType& targetGraph, const std::vector<std::vector<IndexType>>& sourceToTargetLabels, const std::vector<std::vector<std::vector<std::vector<ValueType>>>>& sourceToTargetDistances, IndexType sourceGraphRoot, ValueType beta, std::vector<IndexType>& sourceToTargetOptLabels)
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
    return curLabel != getNumLabels(curNode) ? 0 : numeric_limits<ValueType>::lowest();
  };

  const std::function<ValueType(IndexType, IndexType, IndexType, IndexType)> func2 = [getNthChild, beta, &sourceToTargetDistances](IndexType curNode, IndexType curLabel, IndexType child, IndexType childLabel)
  {
    const IndexType childNode = getNthChild(curNode, child);
    const ValueType distance = sourceToTargetDistances[curNode][curLabel][child][childLabel];

    return (distance < 0) ? (-distance * beta) : distance;
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

  ValueType sourceToTargetGraphsDistance = 0;

  for (IndexType curNode = 0; curNode < numSourceNodes; ++curNode)
  {
    const IndexType curLabel = optLabels[curNode];

    for (IndexType child = 0; child < getNumChildren(curNode); ++child)
    {
      const IndexType childNode = getNthChild(curNode, child);
      const IndexType childLabel = optLabels[childNode];

      SourceGraphEdgeType e;
      tie(e, tuples::ignore) = edge(curNode, childNode, sourceGraph);

      const ValueType distance = sourceToTargetDistances[curNode][curLabel][child][childLabel];
      if (distance > 0)
      {
        sourceToTargetGraphsDistance += distance;
      }
    }
  }

  return sourceToTargetGraphsDistance;
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

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  std::string sourceFileName;
  std::string targetFileName;
  std::string sourcePrimeFileNameMask;
  std::string targetPrimeFileNameMask;

  bool outputPrimeGraphs = false;
  int knn = 7;
  int sourceGraphRoot = 0;

  po::options_description desc;

  desc.add_options()
    ("help", "print usage message")
    ("sourceFileName", po::value(&sourceFileName)->required(), "source filename")
    ("targetFileName", po::value(&targetFileName)->required(), "target filename")
    ("sourcePrime", po::value(&sourcePrimeFileNameMask), "sourcePrime filename mask")
    ("targetPrime", po::value(&targetPrimeFileNameMask), "targetPrime filename mask")
    ("outputPrimeGraphs", "output sourcePrime and targetPrime graphs")
    ("knn", po::value(&knn), "number of nearest neighbors")
    ("sourceRoot", po::value(&sourceGraphRoot), "root node index");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    desc.print(std::cout);
    return EXIT_SUCCESS;
  }

  using namespace std;
  using namespace boost;
  using namespace Eigen;

  constexpr size_t NumDimensions = 3U;

  typedef double ValueType;
  typedef int IndexType;

  typedef Matrix<ValueType, 1U, NumDimensions> PositionType;

  typedef SourceGraphNode<PositionType> SourceGraphNodeType;
  typedef SourceGraphEdge<ValueType> SourceGraphEdgeType;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, SourceGraphNodeType, SourceGraphEdgeType> SourceGraphType;

  BOOST_LOG_TRIVIAL(info) << "source filename = \"" << sourceFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "target filename = \"" << targetFileName << "\"";
  BOOST_LOG_TRIVIAL(info) << "sourcePrime filename mask = \"" << sourcePrimeFileNameMask << "\"";
  BOOST_LOG_TRIVIAL(info) << "targetPrime filename mask = \"" << targetPrimeFileNameMask << "\"";
  BOOST_LOG_TRIVIAL(info) << "number of nearest neighbors = " << knn;

  const string positionsDataSetName = "positions";
  const string measurementsDataSetName = "measurements";
  const string indices1DataSetName = "indices1";
  const string indices2DataSetName = "indices2";
  const string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const string radiusesDataSetName = "radiuses";

  FileReader sourceFileReader(sourceFileName);

  vector<ValueType> sourcePositions;
  vector<IndexType> sourceIndices1;
  vector<IndexType> sourceIndices2;
  vector<ValueType> sourceRadiuses;

  sourceFileReader.Read(positionsDataSetName, sourcePositions);
  sourceFileReader.Read(indices1DataSetName, sourceIndices1);
  sourceFileReader.Read(indices2DataSetName, sourceIndices2);
  sourceFileReader.Read(radiusesDataSetName, sourceRadiuses);

  FileReader targetFileReader(targetFileName);

  //vector<ValueType> targetMeasurements;
  vector<ValueType> targetPositions;
  vector<IndexType> targetIndices1;
  vector<IndexType> targetIndices2;
  //vector<ValueType> targetTangentLinesPoints1;
  //vector<ValueType> targetTangentLinesPoints2;
  //vector<ValueType> targetRadiuses;

  targetFileReader.Read(positionsDataSetName, targetPositions);
  //targetFileReader.Read(measurementsDataSetName, targetMeasurements);
  targetFileReader.Read(indices1DataSetName, targetIndices1);
  targetFileReader.Read(indices2DataSetName, targetIndices2);
  //targetFileReader.Read(tangentLinesPoints1DataSetName, targetTangentLinesPoints1);
  //targetFileReader.Read(tangentLinesPoints2DataSetName, targetTangentLinesPoints2);
  //targetFileReader.Read(radiusesDataSetName, targetRadiuses);

  const auto sourceGraph = GenerateSourceGraph<NumDimensions>(sourcePositions, sourceIndices1, sourceIndices2, sourceRadiuses, sourceGraphRoot);
  const auto targetGraph = GenerateTargetGraph<NumDimensions>(targetPositions, targetIndices1, targetIndices2);

  vector<vector<IndexType>> knnLabels;

  GenerateSourceToTargetKnnLabels<NumDimensions>(sourcePositions, targetPositions, knnLabels, knn);

  vector<vector<vector<vector<ValueType>>>> sourceToTargetDistances;

  GenerateSourceToTargetDistances<NumDimensions>(sourceGraph, targetGraph, knnLabels, sourceToTargetDistances);

  const ValueType sourceGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(sourceGraph);
  const ValueType targetGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(targetGraph);

  constexpr int numBetas = 1000;
  constexpr ValueType maxBetaValue = 50;

  const std::function<ValueType(int)> getKthBeta = [numBetas,maxBetaValue](int zeroBasedIndex)
  {
    return maxBetaValue * (zeroBasedIndex + ValueType(1)) / numBetas;
  };

  vector<ValueType> betas(numBetas);
  vector<ValueType> sourceToTargetGraphsDistances(numBetas);
  vector<ValueType> sourceGraphsLengthRatios(numBetas);
  vector<ValueType> targetGraphsLengthRatios(numBetas);

//#ifndef _MSC_VER
#pragma omp parallel for
//#endif
  for (int k = 0; k < numBetas; ++k)
  {
    const ValueType kthBeta = getKthBeta(k);
    betas[k] = kthBeta;

    vector<IndexType> sourceToTargetOptLabels;
    sourceToTargetGraphsDistances[k] = GenerateSourceToTargetOptLabels(sourceGraph, targetGraph, knnLabels, sourceToTargetDistances, sourceGraphRoot, kthBeta, sourceToTargetOptLabels);

    const auto sourcePrimeGraph = GenerateSourcePrimeGraph(sourceGraph, sourceToTargetOptLabels);
    const ValueType sourcePrimeGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(sourcePrimeGraph);
    sourceGraphsLengthRatios[k] = sourcePrimeGraphLength / sourceGraphLength;

    const auto targetPrimeGraph = GenerateTargetPrimeGraph(sourceGraph, targetGraph, sourceToTargetOptLabels);
    const ValueType targetPrimeGraphLength = ComputeEuclideanLengthOfGraph<ValueType>(targetPrimeGraph);
    targetGraphsLengthRatios[k] = targetPrimeGraphLength / targetGraphLength;
  }

  cout << "kth-beta,source-to-target-graphs-distance,source-graphs-length-ratio,target-graphs-length-ratio" << endl;

  for (int k = 0; k < numBetas; ++k)
  {
    const ValueType kthBeta = betas[k];
    const ValueType sourceToTargetGraphsDistance = sourceToTargetGraphsDistances[k];
    const ValueType sourceGraphsLengthRatio = sourceGraphsLengthRatios[k];
    const ValueType targetGraphsLengthRatio = targetGraphsLengthRatios[k];

    cout << fixed << setprecision(3)
      << kthBeta << ',' << setprecision(5)
      << sourceToTargetGraphsDistance << ','
      << sourceGraphsLengthRatio << ','
      << targetGraphsLengthRatio << endl;
  }

  if (outputPrimeGraphs)
  {
    for (ValueType pct = 10; pct < 100; pct += 10)
    {
      BOOST_LOG_TRIVIAL(info) << "source-graphs-length-ratio = " << pct << "%";

      const ValueType ratio = pct / 100;

      const auto iter = lower_bound(sourceGraphsLengthRatios.cbegin(), sourceGraphsLengthRatios.cend(), ratio);

      if (iter != sourceGraphsLengthRatios.cend())
      {
        const int k = distance(sourceGraphsLengthRatios.cbegin(), iter);

        const ValueType kthBeta = getKthBeta(k);
        betas[k] = kthBeta;

        vector<IndexType> sourceToTargetOptLabels;
        sourceToTargetGraphsDistances[k] = GenerateSourceToTargetOptLabels(sourceGraph, targetGraph, knnLabels, sourceToTargetDistances, sourceGraphRoot, kthBeta, sourceToTargetOptLabels);

        const auto sourcePrimeGraph = GenerateSourcePrimeGraph(sourceGraph, sourceToTargetOptLabels);

        vector<IndexType> sourcePrimeIndices1;
        vector<IndexType> sourcePrimeIndices2;
        vector<ValueType> sourcePrimeRadiusesPrime;
        GenerateSourceGraphDataSet(sourcePrimeGraph, sourcePrimeIndices1, sourcePrimeIndices2, sourcePrimeRadiusesPrime);

        format sourcePrimeFormatter(sourcePrimeFileNameMask);
        sourcePrimeFormatter % pct;

        const string sourcePrimeFileName = sourcePrimeFormatter.str();

        BOOST_LOG_TRIVIAL(info) << "sourcePrimeFileName = " << sourcePrimeFileName;
        BOOST_LOG_TRIVIAL(info) << "sourcePrimeIndices1.size = " << sourcePrimeIndices1.size();
        BOOST_LOG_TRIVIAL(info) << "sourcePrimeIndices2.size = " << sourcePrimeIndices2.size();

        const auto targetPrimeGraph = GenerateTargetPrimeGraph(sourceGraph, targetGraph, sourceToTargetOptLabels);

        vector<IndexType> targetPrimeIndices1;
        vector<IndexType> targetPrimeIndices2;
        GenerateTargetGraphDataSet(targetPrimeGraph, targetPrimeIndices1, targetPrimeIndices2);

        format targetPrimeFormatter(targetPrimeFileNameMask);
        targetPrimeFormatter % pct;

        const string targetPrimeFileName = targetPrimeFormatter.str();

        BOOST_LOG_TRIVIAL(info) << "targetPrimeFileName = " << targetPrimeFileName;
        BOOST_LOG_TRIVIAL(info) << "targetPrimeIndices1.size = " << targetPrimeIndices1.size();
        BOOST_LOG_TRIVIAL(info) << "targetPrimeIndices2.size = " << targetPrimeIndices2.size();

        FileWriter sourcePrimeGraphFileWriter(sourcePrimeFileName);
        FileWriter targetPrimeGraphFileWriter(targetPrimeFileName);

        sourcePrimeGraphFileWriter.Write(positionsDataSetName, sourcePositions);
        sourcePrimeGraphFileWriter.Write(indices1DataSetName, sourcePrimeIndices1);
        sourcePrimeGraphFileWriter.Write(indices2DataSetName, sourcePrimeIndices2);
        sourcePrimeGraphFileWriter.Write(radiusesDataSetName, sourcePrimeRadiusesPrime);

        targetPrimeGraphFileWriter.Write(positionsDataSetName, targetPositions);
        //targetPrimeGraphFileWriter.Write(measurementsDataSetName, targetMeasurements);
        targetPrimeGraphFileWriter.Write(indices1DataSetName, targetPrimeIndices1);
        targetPrimeGraphFileWriter.Write(indices2DataSetName, targetPrimeIndices2);
        //targetPrimeGraphFileWriter.Write(tangentLinesPoints1DataSetName, targetTangentLinesPoints1);
        //targetPrimeGraphFileWriter.Write(tangentLinesPoints2DataSetName, targetTangentLinesPoints2);
        //targetPrimeGraphFileWriter.Write(radiusesDataSetName, targetRadiuses);
      }
    }
  }
}
