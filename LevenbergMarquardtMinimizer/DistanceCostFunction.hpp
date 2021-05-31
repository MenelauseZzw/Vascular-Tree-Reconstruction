#ifndef DistanceCostFunction_hpp
#define DistanceCostFunction_hpp

template<typename ValueType, int NumDimensions>
void CpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void GpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void CpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void GpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength);

#include "DistanceCostFunction.inl"

#endif//DistanceCostFunction_hpp