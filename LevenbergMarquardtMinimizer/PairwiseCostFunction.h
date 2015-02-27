#ifndef PairwiseCostFunction_h
#define PairwiseCostFunction_h

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

struct PairwiseCostFunction : public thrust::binary_function<float, float, float>
{
  struct GradientWithRespectToParams : public thrust::binary_function<float, float, float>
  {

  };
};

#endif//PairwiseCostFunction_h