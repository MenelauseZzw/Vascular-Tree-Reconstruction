#ifndef GraphDataReader_hpp
#define GraphDataReader_hpp

#include <string>

template<typename ValueType, typename IndexType>
struct GraphData;

template<typename ValueType, typename IndexType>
struct GraphDataReader
{
  struct Options
  {
    std::string sourceFileName;
    std::string measurementsDataSetName;
    std::string tangentsLinesPoints1DataSetName;
    std::string tangentsLinesPoints2DataSetName;
    std::string radiusesDataSetName;
  };

  void Read(const Options& readerOptions, GraphData<ValueType, IndexType>& graphData) const;
};

#endif//GraphDataReader_hpp