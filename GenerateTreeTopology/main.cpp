#include "FileReader.hpp"
#include "FileWriter.hpp"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/concept_check.hpp>
#include <boost/operators.hpp>
#include <boost/iterator.hpp>
#include <boost/multi_array.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <exception>
#include <flann/flann.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "edmonds_optimum_branching.hpp"
# define PI 3.14159265358979323846  /* pi */


namespace boost {
	//code from https://github.com/atofigh/edmonds-alg/blob/master/test/test.cpp
	struct complete_graph {
		complete_graph(int n_vertices) : n_vertices(n_vertices) {}
		int n_vertices;

		struct edge_iterator : public input_iterator_helper<edge_iterator, int, std::ptrdiff_t, int const *, int>
		{
			int edge_idx, n_vertices;

			edge_iterator() : edge_idx(0), n_vertices(-1) {}
			edge_iterator(int n_vertices, int edge_idx) : edge_idx(edge_idx), n_vertices(n_vertices) {}
			edge_iterator &operator++()
			{
				if (edge_idx >= n_vertices * n_vertices)
					return *this;
				++edge_idx;
				if (edge_idx / n_vertices == edge_idx % n_vertices)
					++edge_idx;
				return *this;
			}
			int operator*() const { return edge_idx; }
			bool operator==(const edge_iterator &iter) const
			{
				return edge_idx == iter.edge_idx;
			}
		};
	};

	struct knn_graph {
		knn_graph(int n_vertices) : n_vertices(n_vertices), n_edges(0) {}
		int n_vertices, n_edges;
		std::vector<int> indices1;
		std::vector<int> indices2;

		struct edge_iterator : public input_iterator_helper<edge_iterator, int, std::ptrdiff_t, int const *, int>
		{
			int edge_idx, edge_num;

			edge_iterator(int edge_idx, int edge_num) : edge_idx(edge_idx), edge_num(edge_num) {}
			edge_iterator &operator++()
			{
				if (edge_idx >= edge_num)
					return *this;
				++edge_idx;
				return *this;
			}
			int operator*() const { return edge_idx; }
			bool operator==(const edge_iterator &iter) const
			{
				return edge_idx == iter.edge_idx;
			}
		};
	};

	template<>
	struct graph_traits<complete_graph> {
		typedef int                             vertex_descriptor;
		typedef int                             edge_descriptor;
		typedef directed_tag                    directed_category;
		typedef disallow_parallel_edge_tag      edge_parallel_category;
		typedef edge_list_graph_tag             traversal_category;
		typedef complete_graph::edge_iterator   edge_iterator;
		typedef unsigned                        edges_size_type;

		static vertex_descriptor null_vertex() { return -1; }
	};

	template<>
	struct graph_traits<knn_graph> {
		typedef int                             vertex_descriptor;
		typedef int                             edge_descriptor;
		typedef directed_tag                    directed_category;
		typedef disallow_parallel_edge_tag      edge_parallel_category;
		typedef edge_list_graph_tag             traversal_category;
		typedef knn_graph::edge_iterator        edge_iterator;
		typedef unsigned                        edges_size_type;

		static vertex_descriptor null_vertex() { return -1; }
	};

	std::pair<knn_graph::edge_iterator, knn_graph::edge_iterator>
		edges(const knn_graph &g)
	{
		int n_edges = g.n_edges;
		return std::make_pair(knn_graph::edge_iterator(0, n_edges),
			knn_graph::edge_iterator(n_edges - 1, n_edges));
	}

	std::pair<complete_graph::edge_iterator, complete_graph::edge_iterator>
		edges(const complete_graph &g)
	{
		return std::make_pair(complete_graph::edge_iterator(g.n_vertices, 1),
			complete_graph::edge_iterator(g.n_vertices, g.n_vertices*g.n_vertices));
	}

	void 
		add_edge(int source, int target, knn_graph &g)
	{
		g.indices1.push_back(source);
		g.indices2.push_back(target);
		g.n_edges++;
	}

	unsigned
		num_edges(const complete_graph &g)
	{
		return (g.n_vertices - 1) * (g.n_vertices - 1);
	}

	unsigned 
		num_edges(const knn_graph &g)
	{
		return g.n_edges;
	}

	int
		source(int edge, const complete_graph &g)
	{
		return edge / g.n_vertices;
	}

	int
		source(int edge, const knn_graph &g)
	{
		return g.indices1[edge];
	}

	int
		target(int edge, const complete_graph &g)
	{
		return edge % g.n_vertices;
	}

	int
		target(int edge, const knn_graph &g)
	{
		return g.indices2[edge];
	}
}

template<typename ValueType, typename PositionType>
ValueType ComputeEuclideanDistance(
	const PositionType& point1,
	const PositionType& point2,
	const PositionType& tangentLine1Point1,
	const PositionType& tangentLine1Point2,
	const PositionType& tangentLine2Point1,
	const PositionType& tangentLine2Point2,
	ValueType radius1,
	ValueType radius2, ValueType epsilon)
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
ValueType ComputeArcLengthsSumDistance(const PositionType& point1, const PositionType& point2, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& tangentLine2Point1, const PositionType& tangentLine2Point2, ValueType radius1, ValueType radius2, ValueType epsilon)
{
	// stable way to compute the arclength between two points
	const PositionType tangentline1 = tangentLine1Point1 - tangentLine1Point2;
	const PositionType tangentline2 = tangentLine2Point1 - tangentLine2Point2;
	const PositionType baseline = point1 - point2;
	ValueType eps = 0.45;

	const PositionType dis2totangentline1 = baseline - (baseline.dot(tangentline1) / tangentline1.squaredNorm())*tangentline1;
	const ValueType ratio1 = dis2totangentline1.norm() / baseline.norm();
	const ValueType arc1Length = (ratio1 <= eps) ? baseline.norm() + dis2totangentline1.squaredNorm() / (6 * baseline.norm()) : baseline.squaredNorm() / dis2totangentline1.norm()*asin(ratio1);
	//const ValueType arc1Length = (ratio1<=epsilon) ? baseline.norm() : baseline.squaredNorm()/dis2totangentline1.norm()*asin(ratio1); 

	const PositionType dis1totangentline2 = baseline - (baseline.dot(tangentline2) / tangentline2.squaredNorm())*tangentline2;
	const ValueType ratio2 = dis1totangentline2.norm() / baseline.norm();
	const ValueType arc2Length = (ratio2 <= eps) ? baseline.norm() + dis1totangentline2.squaredNorm() / (6 * baseline.norm()) : baseline.squaredNorm() / dis1totangentline2.norm()*asin(ratio2);
	//const ValueType arc2Length = (ratio2<=epsilon) ? baseline.norm() : baseline.squaredNorm()/dis1totangentline2.norm()*asin(ratio2); 

	return arc1Length + arc2Length;
}

template<typename ValueType, typename PositionType>
ValueType ComputeArcLengthsMinDistance(const PositionType& point1, const PositionType& point2, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& tangentLine2Point1, const PositionType& tangentLine2Point2, ValueType radius1, ValueType radius2, ValueType epsilon)
{
	// stable way to compute the arclength between two points
	const PositionType tangentline1 = tangentLine1Point1 - tangentLine1Point2;
	const PositionType tangentline2 = tangentLine2Point1 - tangentLine2Point2;
	const PositionType baseline = point1 - point2;

	ValueType eps = 0.45;
	const PositionType dis2totangentline1 = baseline - (baseline.dot(tangentline1) / tangentline1.squaredNorm())*tangentline1;// the distance "vector" from point2 to tangentline1 
	const ValueType ratio1 = dis2totangentline1.norm() / baseline.norm();
	const ValueType arc1Length = (ratio1 <= eps) ? baseline.norm() + dis2totangentline1.squaredNorm() / (6 * baseline.norm()) : baseline.squaredNorm() / dis2totangentline1.norm()*asin(ratio1);

	const PositionType dis1totangentline2 = baseline - (baseline.dot(tangentline2) / tangentline2.squaredNorm())*tangentline2;
	const ValueType ratio2 = dis1totangentline2.norm() / baseline.norm();
	const ValueType arc2Length = (ratio2 <= eps) ? baseline.norm() + dis1totangentline2.squaredNorm() / (6 * baseline.norm()) : baseline.squaredNorm() / dis1totangentline2.norm()*asin(ratio2);
	return std::min(arc1Length, arc2Length);
}

template<typename ValueType, typename PositionType>
ValueType ComputeDirectedArclength(const PositionType& point1, const PositionType& point2, const PositionType& tangentLine1Point1, const PositionType& tangentLine1Point2, const PositionType& tangentLine2Point1, const PositionType& tangentLine2Point2, ValueType epsilon)
{
	//point1 is p, point2 is q
	//find the direction of lpq
	PositionType lpq;
	PositionType lp = tangentLine1Point2 - tangentLine1Point1;
	lp = lp / lp.norm();
	PositionType baseLine = point2 - point1;
	baseLine = baseLine / baseLine.norm();
	const PositionType mid = (point1 + point2) / 2;


	const ValueType dot1 = baseLine.dot(lp);
	if (dot1 == 0)
	{
		lpq = -1 * lp;
	}
	else
	{
		ValueType t = (baseLine.dot(point1) - baseLine.dot(mid)) / (-1 * dot1 + 1e-15);
		const PositionType lp_ = point1 + t*lp;
		if (t >= 0)
		{
			lpq = point2 - lp_;
		}
		else
		{
			lpq = lp_ - point2;
		}
	}

	//assignment for edge weight
	PositionType lq = tangentLine2Point2 - tangentLine2Point1;
	lq = lq / lq.norm();
	const ValueType dot2 = lq.dot(lpq);
	if (dot2 < 0)
	{
		return 1e+8;
	}
	else
	{
		const ValueType dot3 = lp.dot(baseLine);
		if (1 - dot3 < epsilon)
		{
			return (point2 - point1).norm();
		}
		else if (1 + dot3 < epsilon)
		{
			return 1e+8;
		}
		else
		{
			PositionType arcCenter;
			ValueType arcRadius;

			ComputeArcCenterAndRadius(point1, tangentLine1Point1, tangentLine1Point2, point2, arcCenter, arcRadius);
			ValueType arclength;
			if (dot3 >= 0)
			{
				return arcRadius * acos((point1 - arcCenter).dot(point2 - arcCenter) / (arcRadius * arcRadius));
			}
			else
			{
				return arcRadius * 2 * PI - arcRadius * acos((point1 - arcCenter).dot(point2 - arcCenter) / (arcRadius * arcRadius));
			}
		}
	}
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
	const std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType, ValueType, ValueType)>& distanceFunc,
	const std::vector<ValueType>& positions,
	const std::vector<ValueType>& tangentLinesPoints1,
	const std::vector<ValueType>& tangentLinesPoints2,
	const std::vector<ValueType>& radiuses,
	std::vector<IndexType>& indices1,
	std::vector<IndexType>& indices2,
	std::vector<ValueType>& weights,
	int knn,
	ValueType epsilon,
	bool msf)
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
	
	std::vector<IndexType> newIndices1;
	std::vector<IndexType> newIndices2;

	if (msf)
	{
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

			const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2, epsilon);
			weightmap[e] = distance;
		}

		indices1.clear();
		indices2.clear();
	}
	else
	{
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

				const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2, epsilon);
				weightmap[e] = distance;
				
				newIndices1.push_back(index1);
			    newIndices2.push_back(index2);
			    weights.push_back(distance);
			}

			indices1.clear();
			indices2.clear();
			
			indices1 = newIndices1;
			indices2 = newIndices2;
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

					const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, radius1, radius2, epsilon);
					weightmap[e] = distance;
				}
			}
		}
	}
	
	/*
	// dijkstra testing
	typedef typename graph_traits<GraphType>::vertex_descriptor VertexType;
	std::vector< VertexType > p(num_vertices(origGraph));
    std::vector< ValueType > d(num_vertices(origGraph));
    
    VertexType s = root;
    
	dijkstra_shortest_paths(origGraph, s,
        predecessor_map(boost::make_iterator_property_map(
                            p.begin(), get(boost::vertex_index, origGraph)))
            .distance_map(boost::make_iterator_property_map(
                d.begin(), get(boost::vertex_index, origGraph))));
    
    std::vector<bool> vertex_visited;
    for(int i=0;i<num_vertices(origGraph);i++)
    {
    	vertex_visited.push_back(false);
    }             
    for(int v = 0; v<num_vertices(origGraph);v++)
    {
    	VertexType current = v;
    	while(current!=s)
    	{
    		if(vertex_visited[current]==true) break;
    	    vertex_visited[current] = true;
    		indices1.push_back(current);
    		indices2.push_back(p[current]);
    		current = p[current];
    	}
    }*/
    
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

template<int NumDimensions, typename ValueType, typename IndexType, typename PositionType>
void GenerateMinimumArborescence(
	const std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType)>& distanceFunc,
	const std::vector<ValueType>& positions,
	const std::vector<ValueType>& tangentLinesPoints1,
	const std::vector<ValueType>& tangentLinesPoints2,
	const std::vector<ValueType>& radiuses,
	std::vector<IndexType>& indices1,
	std::vector<IndexType>& indices2,
	std::vector<ValueType>& weights,
	int knn,
	ValueType epsilon,
	int root)
{   
	using namespace std;
	using namespace boost;
	using namespace Eigen;

	typedef adjacency_list<vecS, vecS, bidirectionalS, no_property, property<edge_weight_t, ValueType>> GraphType;
	typedef typename graph_traits<GraphType>::edge_descriptor GraphEdgeType;
	typedef typename graph_traits<GraphType>::vertex_descriptor VertexType;
	typedef typename property_map<GraphType, edge_weight_t>::type WeightMapType;

	typedef Matrix<ValueType, Dynamic, NumDimensions, RowMajor> MatrixType;

	const size_t numberOfPoints = positions.size() / NumDimensions;

	Map<const MatrixType> positions_(positions.data(), numberOfPoints, NumDimensions);
	Map<const MatrixType> tangentLinesPoints1_(tangentLinesPoints1.data(), numberOfPoints, NumDimensions);
	Map<const MatrixType> tangentLinesPoints2_(tangentLinesPoints2.data(), numberOfPoints, NumDimensions);

	GraphType origGraph(numberOfPoints);
	WeightMapType weightmap = get(edge_weight, origGraph);

	vector<int> flag(numberOfPoints, 0);
	vector<vector<int>> flag2d(numberOfPoints, flag);

	GenerateKnnGraph(
		knn,
		positions,
		tangentLinesPoints1,
		tangentLinesPoints2,
		indices1,
		indices2,
		NumDimensions);

	GraphEdgeType e;
	
	std::vector<IndexType> newIndices1;
	std::vector<IndexType> newIndices2;

	for (size_t i = 0; i < indices1.size(); ++i)
	{
		const IndexType index1 = indices1[i];
		const IndexType index2 = indices2[i];

		// from index1 to index2
		if (flag2d[index1][index2] == 0)
		{
			tie(e, tuples::ignore) = add_edge(index1, index2, origGraph);

			const auto& point1 = positions_.row(index1);
			const auto& tangentLine1Point1 = tangentLinesPoints1_.row(index1);
			const auto& tangentLine1Point2 = tangentLinesPoints2_.row(index1);

			const auto& point2 = positions_.row(index2);
			const auto& tangentLine2Point1 = tangentLinesPoints1_.row(index2);
			const auto& tangentLine2Point2 = tangentLinesPoints2_.row(index2);

			const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, epsilon);
			weightmap[e] = distance;

			flag2d[index1][index2] += 1;
			
			newIndices1.push_back(index1);
			newIndices2.push_back(index2);
			weights.push_back(distance);
		}

		// from index2 to index1
		if (flag2d[index2][index1] == 0)
		{
			tie(e, tuples::ignore) = add_edge(index2, index1, origGraph);

			const auto& point1 = positions_.row(index2);
			const auto& tangentLine1Point1 = tangentLinesPoints1_.row(index2);
			const auto& tangentLine1Point2 = tangentLinesPoints2_.row(index2);

			const auto& point2 = positions_.row(index1);
			const auto& tangentLine2Point1 = tangentLinesPoints1_.row(index1);
			const auto& tangentLine2Point2 = tangentLinesPoints2_.row(index1);

			const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, epsilon);
			weightmap[e] = distance;

			flag2d[index2][index1] += 1;
			
			newIndices1.push_back(index2);
			newIndices2.push_back(index1);
			weights.push_back(distance);
		}
	}
    
    
	indices1.clear();
	indices2.clear();
	
	indices1 = newIndices1;
	indices2 = newIndices2;
    
	VertexType roots[] = { 0 };
	roots[0] = root;
	vector<GraphEdgeType> branching;

	edmonds_optimum_branching<false, true, true>(origGraph, identity_property_map(), weightmap, roots, roots + 1, back_inserter(branching));

	BOOST_FOREACH(GraphEdgeType e, branching)
	{
		const IndexType index1 = source(e, origGraph);
		const IndexType index2 = target(e, origGraph);

		indices1.push_back(index1);
		indices2.push_back(index2);
	}

	/*On complete graph
	using namespace std;
	using namespace boost;
	using namespace Eigen;

	typedef Matrix<ValueType, Dynamic, NumDimensions, RowMajor> MatrixType;

	const size_t n_vertices = positions.size() / NumDimensions;

	Map<const MatrixType> positions_(positions.data(), n_vertices, NumDimensions);
	Map<const MatrixType> tangentLinesPoints1_(tangentLinesPoints1.data(), n_vertices, NumDimensions);
	Map<const MatrixType> tangentLinesPoints2_(tangentLinesPoints2.data(), n_vertices, NumDimensions);

	typedef graph_traits<complete_graph>::edge_descriptor Edge;
	typedef graph_traits<complete_graph>::vertex_descriptor Vertex;
	typedef graph_traits<complete_graph>::edge_iterator Edgeiter;

	complete_graph digraph(n_vertices);
	multi_array<ValueType, 2> diweights(extents[n_vertices][n_vertices]);
	Vertex roots[] = { 0 };
	roots[0] = root;
	vector<Edge> branching;

	pair<Edgeiter, Edgeiter> beiter = edges(digraph);
	Edgeiter biter = beiter.first;
	Edgeiter eiter = beiter.second;

	for (Edgeiter ei = biter; ei.edge_idx < eiter.edge_idx; ++ei)
	{
		Edge eid = ei.edge_idx;
		Vertex snode = source(eid, digraph);
		Vertex enode = target(eid, digraph);

		const auto& point1 = positions_.row(snode);
		const auto& tangentLine1Point1 = tangentLinesPoints1_.row(snode);
		const auto& tangentLine1Point2 = tangentLinesPoints2_.row(snode);

		const auto& point2 = positions_.row(enode);
		const auto& tangentLine2Point1 = tangentLinesPoints1_.row(enode);
		const auto& tangentLine2Point2 = tangentLinesPoints2_.row(enode);

		const ValueType distance = distanceFunc(point1, point2, tangentLine1Point1, tangentLine1Point2, tangentLine2Point1, tangentLine2Point2, epsilon);
		
		diweights[snode][enode] = distance;
	}

	edmonds_optimum_branching<false, true, true>(digraph, identity_property_map(), diweights.origin(),roots, roots + 1, back_inserter(branching));

	BOOST_FOREACH(Edge e, branching)
	{
		const IndexType index1 = source(e, digraph);
		const IndexType index2 = target(e, digraph);

		indices1.push_back(index1);
		indices2.push_back(index2);
	}
	*/
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

void DoGenerateMinimumSpanningTree(const std::string& inputFileName,
	const std::string& outputFileName,
	bool noPositionsDataSet,
	DistanceOptions distanceOption,
	int knn,
	double epsilon,
	bool msf)
{
	const int NumDimensions = 3;

	typedef double ValueType;
	typedef int IndexType;

	typedef Eigen::Matrix<ValueType, 1, NumDimensions> PositionType;
	typedef std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType, ValueType, ValueType)> DistanceFunctionType;

	const std::string measurementsDataSetName = "measurements";
	const std::string positionsDataSetName = "positions";
	const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
	const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
	const std::string radiusesDataSetName = "radiuses";
	const std::string objectnessMeasureDataSetName = "objectnessMeasure";

	const std::string indices1DataSetName = "indices1";
	const std::string indices2DataSetName = "indices2";
	
	const std::string weightsDataSetName = "weights";

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
	
	BOOST_LOG_TRIVIAL(info) << "number of vertices = " << radiuses.size();

	const std::vector<ValueType>& points = noPositionsDataSet ? measurements : positions;

	std::vector<IndexType> indices1;
	std::vector<IndexType> indices2;
	std::vector<ValueType> weights;

	if (msf)
	{
		inputFileReader.Read(indices1DataSetName, indices1);
		inputFileReader.Read(indices2DataSetName, indices2);
	}

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
			weights,
			knn, epsilon, msf);
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
			weights,
			knn, epsilon, msf);
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
			weights,
			knn, epsilon, msf);
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
	outputFileWriter.Write(weightsDataSetName, weights);
}

void DoGenerateMinimumArborescence(const std::string& inputFileName,
	const std::string& outputFileName,
	int knn,
	double epsilon,
	int root)
{
	const int NumDimensions = 3;

	typedef double ValueType;
	typedef int IndexType;

	typedef Eigen::Matrix<ValueType, 1, NumDimensions> PositionType;
	typedef std::function<ValueType(const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, const PositionType&, ValueType)> DistanceFunctionType;

	const std::string measurementsDataSetName = "measurements";
	const std::string positionsDataSetName = "positions";
	const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
	const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
	const std::string radiusesDataSetName = "radiuses";
	const std::string objectnessMeasureDataSetName = "objectnessMeasure";

	const std::string indices1DataSetName = "indices1";
	const std::string indices2DataSetName = "indices2";
	
	const std::string weightsDataSetName = "weights";

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
        inputFileReader.Read(positionsDataSetName, positions);
	inputFileReader.Read(tangentLinesPoints1DataSetName, tangentLinesPoints1);
	inputFileReader.Read(tangentLinesPoints2DataSetName, tangentLinesPoints2);
	inputFileReader.Read(radiusesDataSetName, radiuses);
	inputFileReader.Read(objectnessMeasureDataSetName, objectnessMeasure);

	const IndexType rootInd = root;

	const std::vector<ValueType>& points = positions;

	std::vector<IndexType> indices1;
	std::vector<IndexType> indices2;
	std::vector<ValueType> weights;

	//inputFileReader.Read(indices1DataSetName, indices1);
	//inputFileReader.Read(indices2DataSetName, indices2);

	GenerateMinimumArborescence<NumDimensions>(
		(DistanceFunctionType)ComputeDirectedArclength<ValueType, PositionType>,
		points,
		tangentLinesPoints1,
		tangentLinesPoints2,
		radiuses,
		indices1,
		indices2,
		weights,
		knn, epsilon, rootInd); //here we use the root index read from input file

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
	outputFileWriter.Write(weightsDataSetName, weights);
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
	double epsilon = 0.45;
	bool msf = false;
	bool directedLabel = false;
	int root = 0;
	double eps = 1e-5;

	po::options_description desc;

	desc.add_options()
		("help", "print usage message")
		("inputFileName", po::value(&inputFileName)->required(), "the name of the input file")
		("outputFileName", po::value(&outputFileName)->required(), "the name of the output file")
		("noPositions", po::value(&noPositionsDataSet), "indicate that '/positions' dataset is not present in the input file")
		("onlyPositions", po::value(&onlyPositionsDataSet), "indicate that only '/positions' dataset is present in the input file")
		("optionNum", po::value(&optionNum), "the option number of distance function between two points")
		("knn", po::value(&knn), "the number of nearest neighbors to consider (if not specified then use complete graph)")
		("epsilon", po::value(&epsilon), "the threshold to decide whether we should use arc length or the euclidean distance")
		("msf", po::value(&msf), "minimum spanning forest, whether the input graph is connected")
		("directedLabel", po::value(&directedLabel)->required(), "if true, we run the optimum branching algorithm")
		("root", po::value(&root)->required(), "root vertex number for generating arborescence")
		("eps", po::value(&eps), "for controling accuracy");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		desc.print(std::cout);
		return EXIT_SUCCESS;
	}

	if (directedLabel)
	{
		try
		{
			DoGenerateMinimumArborescence(inputFileName, outputFileName, knn, eps, root);
			return EXIT_SUCCESS;
		}
		catch (std::exception& e)
		{
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}
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
		DoGenerateMinimumSpanningTree(inputFileName, outputFileName, noPositionsDataSet, (DistanceOptions)optionNum, knn, epsilon, msf);
		return EXIT_SUCCESS;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}
