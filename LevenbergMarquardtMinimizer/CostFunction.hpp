#ifndef CostFunction_hpp
#define CostFunction_hpp

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>

template<typename IndexType, typename ValueType, typename MemorySpace>
class CostFunction
{
public:
  virtual void ComputeJacobian(const cusp::array1d<ValueType, MemorySpace>& x, cusp::csr_matrix<IndexType, ValueType, MemorySpace>& jacobian) const = 0;
  virtual void ComputeResidual(const cusp::array1d<ValueType, MemorySpace>& x, cusp::array1d<ValueType, MemorySpace>& residual) const = 0;

  virtual int JacobianVectorLength() const = 0;
  virtual int ResidualVectorLength() const = 0;
  virtual int VariableVectorLength() const = 0;

  virtual ~CostFunction() { }
};

#endif//CostFunction_hpp