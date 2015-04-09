#ifndef UnaryCostFunctionAndItsGradientWithRespectToParams_h
#define UnaryCostFunctionAndItsGradientWithRespectToParams_h

extern "C" void UnaryCostFunctionAndItsGradientWithRespectToParams3x6(const float* pTildeP, const float* pS, const float *pT, /*const float *pJacTildeP, const float *pJacS, const float *pJacT,*/ const float *pSigma, float* pUnaryCostFunction, float* pUnaryCostGradient, int numPoints);

#endif//UnaryCostFunctionAndItsGradientWithRespectToParams_h