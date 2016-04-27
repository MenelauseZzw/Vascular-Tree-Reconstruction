#ifndef H5Helper_hpp
#define H5Helper_hpp

#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

template<typename ValueType>
class H5Helper;

template<>
class H5Helper<int>
{
public:
  static inline const PredType& GetDataType() { return PredType::NATIVE_INT; }
};

template<>
class H5Helper<double>
{
public:
  static inline const PredType& GetDataType() { return PredType::NATIVE_DOUBLE; }
};

#endif H5Helper_hpp
