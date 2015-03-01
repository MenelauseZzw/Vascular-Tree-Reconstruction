template<int numDims>
__device__ void ProjectionOntoLineAndItsJacobianAt(const float* tildeP, const float* s, const float* t, const float* jacTildeP, const float* jacS, const float* jacT, float* p, float* jacP)
{
  float sMinusT[numDims];
  float sMinusTSq = 0;
  float sMinusTildeP[numDims];
  float jacSMinusT[numDims];

  for (int i = 0; i < numDims; ++i)
  {
    sMinusT[i] = s[i] - t[i];
    sMinusTSq += sMinusT[i] * sMinusT[i];
    sMinusTildeP[i] = s[i] - tildeP[i];
    jacSMinusT[i] = jacS[i] - jacT[i];
  }

  float lambda = 0;
  float nablaSDotSMinusT = 0;
  float nablaTildePDotSMinusT = 0;
  float nablaSMinusTSq = 0;

  for (int i = 0; i < numDims; ++i)
  {
    lambda += (sMinusTildeP[i] * sMinusT[i]) / sMinusTSq;
    nablaSDotSMinusT += sMinusT[i] * jacS[i];
    nablaSDotSMinusT += s[i] * jacSMinusT[i];
    nablaTildePDotSMinusT += sMinusT[i] * jacTildeP[i];
    nablaTildePDotSMinusT += tildeP[i] * jacSMinusT[i];
    nablaSMinusTSq += 2 * sMinusT[i] * jacSMinusT[i];
  }

  float nablaLambda = (nablaSDotSMinusT - lambda * nablaSMinusTSq - nablaTildePDotSMinusT) / sMinusTSq;

  for (int i = 0; i < numDims; ++i)
  {
    p[i] = s[i] - lambda * sMinusT[i];
    jacP[i] = jacS[i] - lambda * jacSMinusT[i] - sMinusT[i] * nablaLambda;
  }
}