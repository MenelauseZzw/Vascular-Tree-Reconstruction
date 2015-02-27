#ifndef UnaryCostFunction_h
#define UnaryCostFunction_h

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <device_types.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <iostream>

typedef thrust::tuple<float, float, float> Coordinates;
typedef thrust::tuple<float, float, float, float, float, float, float> Parameters;

//struct UnaryCostFunction : public thrust::binary_function<Coordinates, Coordinates, float>
struct UnaryCostFunction : public thrust::unary_function<float, float>
{
  struct GradientWithRespectToParams : public thrust::unary_function<float, float>
  {

  };

  //__host__ __device__ float operator()(Coordinates const& coords, Parameters const& params) const
  //{
  //  float3 _p = make_float3(thrust::get<0>(coords), thrust::get<1>(coords), thrust::get<2>(coords));

  //  float3 s = make_float3(thrust::get<0>(params), thrust::get<1>(params), thrust::get<2>(params));
  //  float3 t = make_float3(thrust::get<3>(params), thrust::get<4>(params), thrust::get<5>(params));
  //  float sigma = thrust::get<6>(params);

  //  float3 dir = normalize(t - s);
  //  
  //  float a = dot(_p - s, dir);

  //  //lerp(s, t, );
  //  float3 p = s + dir * a;
  //  dir = p - _p;

  //  return dot(dir, dir) / (sigma * sigma);
  //}
};

#endif//UnaryCostFunction_h