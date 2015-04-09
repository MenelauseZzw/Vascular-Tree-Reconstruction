#ifndef ProjectionOntoLineAndItsJacobian_h
#define ProjectionOntoLineAndItsJacobian_h

extern "C" void ProjectionOntoLineAndItsJacobian3x6(const float* pTildeP, const float* pS, const float *pT, /*const float *pJacTildeP, const float *pJacS, const float *pJacT,*/ float *pP, float* pJacP, unsigned int numPoints);

#endif//ProjectionOntoLineAndItsJacobian_h