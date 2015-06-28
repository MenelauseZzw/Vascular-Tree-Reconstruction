#ifndef DistanceCostFunction_hpp
#define DistanceCostFunction_hpp

template<typename ValueType, int NumDimensions>
void cpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void gpuDistanceCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pResidual, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void cpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, int residualVectorLength);

template<typename ValueType, int NumDimensions>
void gpuDistanceCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType* pJacobian, int residualVectorLength);

#include "DistanceCostFunction.inl"

#endif//DistanceCostFunction_hpp