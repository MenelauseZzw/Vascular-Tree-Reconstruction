#ifndef DoLevenbergMarquardtMinimizer_hpp
#define DoLevenbergMarquardtMinimizer_hpp

#include <vector>

template<int NumDimensions, typename ValueType, typename IndexType>
void DoCpuLevenbergMarquardtMinimizer(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType>& tangentLinesPoints1,
  std::vector<ValueType>& tangentLinesPoints2,
  std::vector<ValueType> const& radiuses,
  std::vector<IndexType> const& indices1,
  std::vector<IndexType> const& indices2,
  std::vector<ValueType> const& lambdas,
  int maxNumberOfIterations,
  double voxelPhysicalSize);

template<int NumDimensions, typename ValueType, typename IndexType>
void DoGpuLevenbergMarquardtMinimizer(
  std::vector<ValueType> const& measurements,
  std::vector<ValueType>& tangentLinesPoints1,
  std::vector<ValueType>& tangentLinesPoints2,
  std::vector<ValueType> const& radiuses,
  std::vector<IndexType> const& indices1,
  std::vector<IndexType> const& indices2,
  std::vector<ValueType> const& lambdas,
  int maxNumberOfIterations,
  double voxelPhysicalSize);

#endif//DoLevenbergMarquardtMinimizer_hpp