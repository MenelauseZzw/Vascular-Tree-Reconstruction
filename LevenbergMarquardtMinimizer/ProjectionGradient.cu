#include <cuda.h>

__global__ void ProjectionGradient(float3 p, float3 s, float3 s_t, float* Jp, const float* Js, const float* Js_t/*, float* outerProduct*/,float* JpRes)
{
  const int ndims = 3;
  int j = threadIdx.x;

  float nablas__s_t_j;
  float nablap__s_t_j;
  float nabla_s_t2_j;

  float s_t2;
  for (int i = 0; i < ndims; ++i)
  {
    s_t2 += s_t[i] * s_t[i];
  }

  float lambda;
  float lambdaNum;
  for (int i = 0; i < ndims; ++i)
  {
    lambdaNum += (s[i] - p[i]) * s_t[i];
  }
  lambda = lambdaNum / s_t2;

  for (int i = 0; i < ndims; ++i)
  {
    nablas__s_t_j += s_t[i] * Js[i * j] + s[i] * Js_t[i * j];
    nablap__s_t_j += s_t[i] * Jp[i * j] + p[i] * Js_t[i * j];
    nabla_s_t2_j += 2 * s_t[i] * Js_t[i * j];
  }

  float nablaLambda_j;

  nablaLambda_j = (nablas__s_t_j - lambda * nabla_s_t2_j - nablap__s_t_j) / s_t2;

  /*for (int i = 0; i < ndims; ++i)
  {
    outerProduct[i * j] = s_t[i] * nablaLambda[j];
  }*/

  for (int i = 0; i < ndims; ++i)
  {
    JpRes[i * j] = Js[i * j] - lambda * Js_t[i * j] - /*outerProduct[i * j]*/s_t[i] * nablaLambda_j;
  }
}