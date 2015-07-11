#ifndef Minimize_hpp
#define Minimize_hpp

//void Minimize(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP, int maxIterations);

template<typename ValueType, typename IndexType>
void Minimize(const ValueType* pMeasurements, ValueType* pTangentLinesPoints1, ValueType* pTangentLinesPoints2, const ValueType* pRadiuses, int numMeasurements, const IndexType* pIndices1, const IndexType* pIndices2, int indicesLength, ValueType* pPositions, int itnlim, ValueType* pDistances, ValueType* pCurvatures);

#endif//Minimize_hpp