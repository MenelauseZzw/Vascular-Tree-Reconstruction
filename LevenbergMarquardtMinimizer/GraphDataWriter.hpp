#ifndef GraphDataWriter_hpp
#define GraphDataWriter_hpp

#include <string>

template<typename ValueType, typename IndexType>
struct GraphData;

template<typename ValueType, typename IndexType>
struct GraphDataWriter
{
  struct Options
  {
    std::string targetFileName;
    std::string measurementsDataSetName;
    std::string tangentsLines1PointsDataSetName;
    std::string tangentsLines2PointsDataSetName;
    std::string radiusesDataSetName;
    std::string positionsDataSetName;
    std::string sourcesDataSetName;
    std::string targetsDataSetName;
  };

  void Write(const Options& writerOptions, const GraphData<ValueType, IndexType>& graphData) const;
};

#endif//GraphDataWriter_hpp