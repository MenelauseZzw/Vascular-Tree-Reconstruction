#ifndef DotProduct_cuh
#define DotProduct_cuh

#include <host_defines.h>
#include <type_traits>

template<typename Arg1ValueType, typename Arg2ValueType, int NumDimensions, typename ResultType = typename std::common_type<Arg1ValueType, Arg2ValueType>::type>
__forceinline__ __host__ __device__ ResultType DotProduct(const Arg1ValueType(&s)[NumDimensions], const Arg2ValueType(&t)[NumDimensions])
{
  ResultType val{};

  for (int i = 0; i != NumDimensions; ++i)
  {
    val += s[i] * t[i];
  }

  return val;
}

#endif//DotProduct_cuh