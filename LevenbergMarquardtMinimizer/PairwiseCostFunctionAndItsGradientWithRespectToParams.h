#ifndef PairwiseCostFunctionAndItsGradientWithRespectToParams_h
#define PairwiseCostFunctionAndItsGradientWithRespectToParams_h

extern "C" void PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(const float* pTildePi, const float* pSi, const float *pTi, const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float* pTildePj, const float* pSj, const float *pTj, const float *pJacTildePj, const float *pJacSj, const float *pJacTj, float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj, int numPoints);
extern "C" void PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations3x12(const float* pTildeP, const float* pS, const float *pT, const int *pIndPi, const int *pIndPj, /*const float *pJacTildePi, const float *pJacSi, const float *pJacTi, const float *pJacTildePj, const float *pJacSj, const float *pJacTj,*/ float* pPairwiseCostFunctioni, float* pPairwiseCostFunctionj, float* pPairwiseCostGradienti, float* pPairwiseCostGradientj, int numPoints);

#endif//PairwiseCostFunctionAndItsGradientWithRespectToParams_h