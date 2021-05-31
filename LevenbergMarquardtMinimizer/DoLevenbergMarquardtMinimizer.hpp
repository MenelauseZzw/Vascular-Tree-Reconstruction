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
	std::vector<ValueType> const& betas,
	int maxNumberOfIterations,
	double voxelPhysicalSize,
	std::vector<ValueType>& costValue,
	double tau);

template<int NumDimensions, typename ValueType, typename IndexType>
void DoGpuLevenbergMarquardtMinimizer(
	std::vector<ValueType> const& measurements,
	std::vector<ValueType>& tangentLinesPoints1,
	std::vector<ValueType>& tangentLinesPoints2,
	std::vector<ValueType> const& radiuses,
	std::vector<IndexType> const& indices1,
	std::vector<IndexType> const& indices2,
	std::vector<ValueType> const& lambdas,
	std::vector<ValueType> const& betas,
	int maxNumberOfIterations,
	double voxelPhysicalSize,
	std::vector<ValueType>& costValue,
	double tau);

#endif//DoLevenbergMarquardtMinimizer_hpp
