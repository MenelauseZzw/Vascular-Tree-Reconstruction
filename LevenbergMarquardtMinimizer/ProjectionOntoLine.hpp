#ifndef ProjectionOntoLine_hpp
#define ProjectionOntoLine_hpp

template<typename ValueType, int NumDimensions>
void CpuProjectionOntoLine(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType* pProjections, int numMeasurements);

template<typename ValueType, int NumDimensions>
void GpuProjectionOntoLine(ValueType const* pMeasurements, ValueType const* pTangentLinesPoints1, ValueType const* pTangentLinesPoints2, ValueType* pProjections, int numMeasurements);

#include "ProjectionOntoLine.inl"

#endif//ProjectionOntoLine_hpp
