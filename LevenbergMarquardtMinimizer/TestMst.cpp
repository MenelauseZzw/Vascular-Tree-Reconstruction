#include "TestMst.h"
#include <algorithm>
#include <boost/foreach.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <cmath>
#include <iostream>
#include <map>

void testMst(float* pP, float *pSigma, float* pS, float* pT, std::vector<int>& indPi, std::vector<int>& indPj, int numPoints, std::vector<int>& indC)
{
  using namespace boost;
  typedef adjacency_list < vecS, vecS, undirectedS,
    property<vertex_distance_t, int>, property < edge_weight_t, float > > Graph;
  typedef std::pair < int, int >E;

  Graph g(numPoints);
  property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);

  std::vector<float> weights;
  for (int k = 0; k < indPi.size(); ++k)
  {
    int i = indPi[k];
    int j = indPj[k];

    float piMinusPj[3];
    float siMinusTi[3];
    float sjMinusTj[3];

    for (int n = 0; n < 3; ++n)
    {
      piMinusPj[n] = pP[3 * i + n] - pP[3 * j + n];
      siMinusTi[n] = pS[3 * i + n] - pT[3 * i + n];
      sjMinusTj[n] = pS[3 * j + n] - pT[3 * j + n];
    }

    //float siMinusTiCrossSjMinusTj[3];

    //siMinusTiCrossSjMinusTj[0] = siMinusTi[1] * sjMinusTj[2] - siMinusTi[2] * sjMinusTj[1]; //s[1] = u[2]v[3] - u[3]v[2]
    //siMinusTiCrossSjMinusTj[1] = siMinusTi[2] * sjMinusTj[0] - siMinusTi[0] * sjMinusTj[2]; //s[2] = u[3]v[1] - u[1]v[3]
    //siMinusTiCrossSjMinusTj[2] = siMinusTi[0] * sjMinusTj[1] - siMinusTi[1] * sjMinusTj[0]; //s[3] = u[1]v[2] - u[2]v[1]

    //float siMinusTiCrossSjMinusTjSq = 0;
    //for (int n = 0; n < 3; ++n)
    //{
    //siMinusTiCrossSjMinusTjSq += siMinusTiCrossSjMinusTj[n] * siMinusTiCrossSjMinusTj[n];
    //}

    //float invSiMinusTiCrossSjMinusTj = 1 / sqrt(siMinusTiCrossSjMinusTjSq);

    //float distanceBetweenLines = 0;
    //for (int n = 0; n < 3; ++n)
    //{
    //distanceBetweenLines += piMinusPj[n] * siMinusTiCrossSjMinusTj[n];
    //}

    //distanceBetweenLines = abs(distanceBetweenLines);
    //distanceBetweenLines *= invSiMinusTiCrossSjMinusTj;

    float piMinusPjSq = 0;
    for (int n = 0; n < 3; ++n)
    {
      piMinusPjSq += piMinusPj[n] * piMinusPj[n];
    }

    float distanceBetweenPoints = std::sqrt(piMinusPjSq);
    weights.push_back(distanceBetweenPoints);
    //weights.push_back(distanceBetweenLines / (pSigma[i] + pSigma[j]));
  }

  for (int k = 0; k < indPi.size(); ++k)
  {
    graph_traits<Graph>::edge_descriptor e; bool inserted;
    boost::tie(e, inserted) = add_edge(indPi[k], indPj[k], g);
    weightmap[e] = weights[k];
  }

  std::cout << "Number of vertices " << num_vertices(g) << std::endl;
  std::cout << "Number of edges " << num_edges(g) << std::endl;

  //prim_minimum_spanning_tree(g, *vertices(g).first, &p[0], distance, weightmap, indexmap, default_dijkstra_visitor());

  std::vector<graph_traits<Graph>::edge_descriptor> spanning_tree;
  kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));
  std::cout << "Number of edges after " << spanning_tree.size() << std::endl;

  indPi.resize(spanning_tree.size());
  indPj.resize(spanning_tree.size());

  typedef graph_traits<Graph>::vertex_descriptor Vertex;
  typedef graph_traits<Graph>::vertices_size_type VertexIndex;

  Graph graph(numPoints);

  std::vector<VertexIndex> rank(num_vertices(graph));
  std::vector<Vertex> parent(num_vertices(graph));

  typedef VertexIndex* Rank;
  typedef Vertex* Parent;

  disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]);

  initialize_incremental_components(graph, ds);
  incremental_components(graph, ds);

  for (int k = 0; k < indPi.size(); ++k)
  {
    indPi[k] = spanning_tree[k].m_source;
    indPj[k] = spanning_tree[k].m_target;

    graph_traits<Graph>::edge_descriptor e; bool inserted;
    boost::tie(e, inserted) = add_edge(indPi[k], indPj[k], graph);
    ds.union_set(indPi[k], indPj[k]);
  }

  typedef component_index<VertexIndex> Components;

  // NOTE: Because we're using vecS for the graph type, we're
  // effectively using identity_property_map for a vertex index map.
  // If we were to use listS instead, the index map would need to be
  // explicitly passed to the component_index constructor.
  Components components(parent.begin(), parent.end());

  indC.resize(numPoints);
  // Iterate through the component indices
  BOOST_FOREACH(VertexIndex current_index, components) {
    //std::cout << "component " << current_index << " contains: ";

    int count = 0;
    // Iterate through the child vertex indices for [current_index]
    BOOST_FOREACH(VertexIndex child_index,
      components[current_index]) {

      indC[child_index] = current_index;
    }
  }
}