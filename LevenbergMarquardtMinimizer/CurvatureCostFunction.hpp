#ifndef CurvatureCostFunction_hpp
#define CurvatureCostFunction_hpp

template<typename ValueType, typename IndexType, int NumDimensions>
void cpuCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength);

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuCurvatureCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength);

template<typename ValueType, typename IndexType, int NumDimensions>
void cpuCurvatureCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength);

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuCurvatureCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength);

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuPairwiseCostJacobian(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pJacobian, int residualVectorLength, double tau);

template<typename ValueType, typename IndexType, int NumDimensions>
void gpuPairwiseCostResidual(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType const* pWeights, ValueType const* pWeights3, IndexType const* pPointsIndexes1, IndexType const* pPointsIndexes2, ValueType* pResidual, int residualVectorLength, double tau);

#include "CurvatureCostFunction.inl"

#endif//CurvatureCostFunction_hpp
