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

typedef thrust::tuple<float, float, float> Coordinates;
typedef thrust::tuple<float, float, float, float, float, float> Parameters;

struct UnaryCostFunction : public thrust::binary_function<Coordinates, Parameters, Parameters>
{
  __host__ __device__ Parameters operator()(Coordinates const& coords, Parameters const& params) const
  {
    float3 p = make_float3(thrust::get<0>(coords), thrust::get<1>(coords), thrust::get<2>(coords));
    float3 s = make_float3(thrust::get<0>(params), thrust::get<1>(params), thrust::get<2>(params));
    float3 t = make_float3(thrust::get<3>(params), thrust::get<4>(params), thrust::get<5>(params));

    dot(s, t);
    return Parameters(s.x, s.y, s.z, t.x, t.y, t.z);
  }
};

#endif//UnaryCostFunction_h