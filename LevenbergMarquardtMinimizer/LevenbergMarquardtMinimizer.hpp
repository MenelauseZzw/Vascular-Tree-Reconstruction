#ifndef LevenbergMarquardtMinimizer_hpp
#define LevenbergMarquardtMinimizer_hpp

#include "CostFunction.hpp"
#include "LSQR.hpp"
#include <cusp/array1d.h>
#include <cusp/blas/blas.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>

template<typename IndexType, typename ValueType, typename MemorySpace>
void LevenbergMarquardtMinimizer(const CostFunction<IndexType, ValueType, MemorySpace>& func, cusp::array1d<ValueType, MemorySpace>& x, float damp, float dampmin)
{
  typedef cusp::array1d<ValueType, MemorySpace> ArrayType;
  typedef cusp::csr_matrix<IndexType, ValueType, MemorySpace> CsrMatrix;
 
  // Local constants
  const ValueType tolx = 1e-6;
  const ValueType tolf = 1e-6;
  const ValueType tolg = 1e-5;

  const ValueType E = 2;
  const ValueType D = 0.5;

  const ValueType pi1 = 0.01;
  const ValueType pi2 = 0.75;

  // Local variables
  int itn = 0;

  CsrMatrix jacobian;
  func.ComputeJacobian(x, jacobian);
  
  CsrMatrix jacobiant(jacobian.num_cols, jacobian.num_rows, jacobian.num_entries);

  ArrayType gradient(jacobian.num_cols);

  ArrayType residualx(jacobian.num_rows);
  ArrayType residualxpy(jacobian.num_rows);

  ArrayType y(jacobian.num_cols);
  ArrayType xpy(jacobian.num_cols);

  cusp::transpose(jacobian, jacobiant);

  func.ComputeResidual(x, residualx);
  ValueType normSqResidualx = cusp::blas::dot(residualx, residualx);

  while (true)
  {
    ValueType rho;
    ValueType normSqResidualxpy;

    {
      // Local constants
      const ValueType atol = 0;
      const ValueType btol = 0;
      const ValueType conlim = 0;
      const int itnlim = 5000;

      // Local variables
      ValueType Anorm, Acond, rnorm, Arnorm, xnorm;
      int istop, itn;

      LSQR(jacobian, jacobiant, residualx, damp, y, atol, btol, conlim, itnlim, istop, itn, Anorm, Acond, rnorm, Arnorm, xnorm);
    
      cusp::blas::axpby(x, y, xpy, 1, -1);
      func.ComputeResidual(xpy, residualxpy);

      normSqResidualxpy = cusp::blas::dot(residualxpy, residualxpy);
      rho = (normSqResidualx - normSqResidualxpy) / (normSqResidualx - rnorm * rnorm);
    }

    if (rho < pi1)
    {
      damp = damp ? (E * damp) : dampmin;
    }
    else
    {
      std::swap(xpy, x);

      bool convergence = false;
      
      if (false)
      {
        convergence = true;
        std::cout << "x-convergence criterion is signalled" << std::endl;
      }

      if ((normSqResidualx - normSqResidualxpy) / normSqResidualxpy <= tolf)
      {
        convergence = true;
        std::cout << "Function convergence criterion is signalled" << std::endl;
      }

      cusp::multiply(jacobiant, residualx, gradient);

      if (cusp::blas::nrm2(gradient) <= tolg)
      {
        convergence = true;
        std::cout << "Gradient convergence criterion is signalled" << std::endl;
      }

      if (convergence) break;

      std::swap(residualxpy, residualx);
      normSqResidualx = normSqResidualx;

      func.ComputeJacobian(x, jacobian.values);
      cusp::transpose(jacobian, jacobiant);

      itn = itn + 1;

      if (rho > pi2) damp = D * damp;
      if (damp < dampmin) damp = 0;
    }
  }
}

#endif//LevenbergMarquardtMinimizer_hpp