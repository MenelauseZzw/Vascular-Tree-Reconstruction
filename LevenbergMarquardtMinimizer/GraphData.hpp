#ifndef GraphData_hpp
#define GraphData_hpp

#include <thrust/host_vector.h>

template<typename ValueType, typename IndexType>
struct GraphData
{
  thrust::host_vector<ValueType> measurements;
  thrust::host_vector<ValueType> tangentsLinesPoints1;
  thrust::host_vector<ValueType> tangentsLinesPoints2;
  thrust::host_vector<ValueType> radiuses;
  thrust::host_vector<ValueType> positions;
  thrust::host_vector<IndexType> sources;
  thrust::host_vector<IndexType> targets;

  void Swap(GraphData<ValueType, IndexType>& other)
  {
    measurements.swap(other.measurements);
    tangentsLinesPoints1.swap(other.tangentsLinesPoints1);
    tangentsLinesPoints2.swap(other.tangentsLinesPoints2);
    radiuses.swap(other.radiuses);
    positions.swap(other.positions);
    sources.swap(other.sources);
    targets.swap(other.targets);
  }
};

#endif//GraphData_hpp