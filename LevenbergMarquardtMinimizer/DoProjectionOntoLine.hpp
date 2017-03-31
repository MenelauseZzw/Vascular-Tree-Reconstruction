#ifndef DoProjectionOntoLine_hpp
#define DoProjectionOntoLine_hpp

#include <vector>

template<int NumDimensions, typename ValueType>
void DoCpuProjectionOntoLine(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType> const& tangentLinesPoints1,
  std::vector<ValueType> const& tangentLinesPoints2,
  std::vector<ValueType>& positions);

template<int NumDimensions, typename ValueType>
void DoGpuProjectionOntoLine(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType> const& tangentLinesPoints1,
  std::vector<ValueType> const& tangentLinesPoints2,
  std::vector<ValueType>& positions);

#endif//DoProjectionOntoLine_hpp