#ifndef DualNumber_cuh
#define DualNumber_cuh

#include <cuda_runtime.h>

template<typename ValueType>
struct __align__(8) DualNumber
{
  ValueType s;//the real part
  ValueType sPrime;//the dual part

  __inline__ __host__ __device__ DualNumber(ValueType s = ValueType(), ValueType sPrime = ValueType())
    : s(s), sPrime(sPrime)
  {
  }

  // =
  __inline__ __host__ __device__ DualNumber<ValueType>& operator=(const DualNumber<ValueType>& a)
  {
    s = a.s;
    sPrime = a.sPrime;
    return *this;
  }

  // = with a scalar
  __inline__ __host__ __device__ DualNumber<ValueType>& operator=(ValueType scal)
  {
    s = scal;
    sPrime = ValueType();
    return *this;
  }

  // += 
  __inline__ __host__ __device__ DualNumber<ValueType>& operator+=(const DualNumber<ValueType>& a)
  {
    sPrime += a.sPrime;
    s += a.s;
    return *this;
  }

  // += with a scalar
  __inline__ __host__ __device__ DualNumber<ValueType>& operator+=(ValueType scal)
  {
    s += scal;
    return *this;
  }

  // -=
  __inline__ __host__ __device__ DualNumber<ValueType>& operator-=(const DualNumber<ValueType>& a)
  {
    sPrime -= a.sPrime;
    s -= a.s;
    return *this;
  }

  // -= with a scalar
  __inline__ __host__ __device__ DualNumber<ValueType>& operator-=(ValueType scal)
  {
    s -= scal;
    return *this;
  }

  // *=
  __inline__ __host__ __device__ DualNumber<ValueType>& operator*=(const DualNumber<ValueType>& a)
  {
    sPrime = sPrime * a.s + s * a.sPrime;
    s *= a.s;
    return *this;
  }

  // *= with a scalar
  __inline__ __host__ __device__ DualNumber<ValueType>& operator*=(ValueType scal)
  {
    sPrime *= scal;
    s *= scal;
    return *this;
  }

  // /=
  __inline__ __host__ __device__ DualNumber<ValueType>& operator/=(const DualNumber<ValueType>& a)
  {
    sPrime = (sPrime - s * a.sPrime / a.s) / a.s;
    s /= a.s;
    return *this;
  }

  // /= with a scalar
  __inline__ __host__ __device__ DualNumber<ValueType>& operator/=(ValueType scal)
  {
    sPrime /= scal;
    s /= scal;
    return *this;
  }
};

template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> MakeDualNumber(ValueType s, ValueType sPrime)
{
  return DualNumber<ValueType>(s, sPrime);
}

// Unary +
template<typename ValueType>
static __inline__ __host__ __device__ const DualNumber<ValueType>& operator+(const DualNumber<ValueType>& a)
{
  return a;
}

// Unary -
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator-(const DualNumber<ValueType>& a)
{
  return MakeDualNumber(-a.s, -a.sPrime);
}

// Binary +
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator+(const DualNumber<ValueType>& a, const DualNumber<ValueType>& b)
{
  return MakeDualNumber(a.s + b.s, a.sPrime + b.sPrime);
}

// Binary + with a scalar: a + scal
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator+(const DualNumber<ValueType>& a, ValueType scal)
{
  return MakeDualNumber(a.s + scal, a.sPrime);
}

// Binary + with a scalar: scal + a
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator+(ValueType scal, const DualNumber<ValueType>& a)
{
  return MakeDualNumber(scal + a.s, a.sPrime);
}

// Binary -
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator-(const DualNumber<ValueType>& a, const DualNumber<ValueType>& b)
{
  return MakeDualNumber(a.s - b.s, a.sPrime - b.sPrime);
}

// Binary - with a scalar: a - scal
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator-(const DualNumber<ValueType>& a, ValueType scal)
{
  return MakeDualNumber(a.s - scal, a.sPrime);
}

// Binary - with a scalar: scal - a
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator-(ValueType scal, const DualNumber<ValueType>& a)
{
  return MakeDualNumber(scal - a.s, -a.sPrime);
}

// Binary *
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator*(const DualNumber<ValueType>& a, const DualNumber<ValueType>& b)
{
  return MakeDualNumber(a.s * b.s, a.sPrime * b.s + a.s * b.sPrime);
}

// Binary * with a scalar: a * scal
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator*(const DualNumber<ValueType>& a, ValueType scal)
{
  return MakeDualNumber(a.s * scal, a.sPrime * scal);
}

// Binary * with a scalar: scal * a
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator*(ValueType scal, const DualNumber<ValueType>& a)
{
  return MakeDualNumber(scal * a.s, scal * a.sPrime);
}

// Binary /
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator/(const DualNumber<ValueType>& a, const DualNumber<ValueType>& b)
{
  return MakeDualNumber(a.s / b.s, (a.sPrime - a.s * b.sPrime / b.s) / b.s);
}

// Binary / with a scalar: a / scal
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator/(const DualNumber<ValueType>& a, ValueType scal)
{
  return MakeDualNumber(a.s / scal, a.sPrime / scal);
}

// Binary / with a scalar: scal / a
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> operator/(ValueType scal, const DualNumber<ValueType>& a)
{
  return MakeDualNumber(scal / a.s, -scal * a.sPrime / (a.s * a.s));
}

// sqrt(s + s') = sqrt(s) + s' / (2 sqrt(s))
template<typename ValueType>
static __inline__ __host__ __device__ DualNumber<ValueType> sqrt(const DualNumber<ValueType>& a)
{
  return MakeDualNumber(std::sqrt(a.s), a.sPrime / (2 * std::sqrt(a.s)));
}

#endif//DualNumber_cuh