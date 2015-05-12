#include "TestLevenbergMarquardtMinimizer1.h"
#include "PairwiseCostFunction.hpp"
#include "PairwiseCostGradientWithRespectToParams.hpp"
#include "ProjectionOntoLine.hpp"
#include "SparseLeastSquares.h"
#include "UnaryCostFunction.hpp"
#include "UnaryCostGradientWithRespectToParams.hpp"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/blas/blas.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <cusp/elementwise.h>
#include <cusp/krylov/cg_m.h>
#include <cusp/linear_operator.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>
#include <iostream>
#include <thrust/device_ptr.h>

void testLevenbergMarquardtMinimizer1(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP, int maxIterations)
{
  const int numDimensions = 3;

  typedef cusp::array1d_view<float*> array1d_view;
  typedef cusp::array2d_view<array1d_view> array2d_view;

  array2d_view hTildeP(numPoints, numDimensions, numDimensions, array1d_view(pTildeP, pTildeP + numDimensions * numPoints));
  array2d_view hS(numPoints, numDimensions, numDimensions, array1d_view(pS, pS + numDimensions * numPoints));
  array2d_view hT(numPoints, numDimensions, numDimensions, array1d_view(pT, pT + numDimensions * numPoints));
  array2d_view hP(numPoints, numDimensions, numDimensions, array1d_view(pP, pP + numDimensions * numPoints));
  array1d_view hSigma(pSigma, pSigma + numPoints);

  const int numParams = 6;
  const int numParamsPlusNumParams = numParams + numParams;

  typedef cusp::array2d<float, cusp::host_memory> array2d;

  cusp::array1d_view<int*> hIndPi(pIndPi, pIndPi + numPairs);
  cusp::array1d_view<int*> hIndPj(pIndPj, pIndPj + numPairs);

  cusp::array2d<float, cusp::device_memory> tildeP(hTildeP);
  cusp::array2d<float, cusp::device_memory> p(numPoints, numDimensions);

  cusp::array1d<float, cusp::host_memory> hBeta(numPoints);

  for (int i = 0; i < hBeta.size(); ++i)
  {
    hBeta[i] = 1 / hSigma[i];
  }

  cusp::array1d<float, cusp::device_memory> beta(hBeta);
  cusp::array1d<float, cusp::device_memory> gamma(numPairs, 20);

  cusp::array1d<int, cusp::device_memory> indPi(hIndPi);
  cusp::array1d<int, cusp::device_memory> indPj(hIndPj);

  cusp::array1d<float, cusp::device_memory> edBar(numPoints);
  cusp::array1d<float, cusp::device_memory> esBar(numPairs);

  cusp::array1d<float, cusp::device_memory> e_(numPoints + numPairs, 0);

  typedef cusp::array1d<float, cusp::device_memory>::view array1d_device_view;
  array1d_device_view e(
    e_.begin(),
    e_.begin() + numPoints
    );

  typedef cusp::csr_matrix<int, float, cusp::device_memory> csr_matrix;

  const int n1 = numPoints * numParams;
  const int n2 = numPairs * numParamsPlusNumParams;

  csr_matrix jacE_(numPoints + numPairs, numPoints * numParams, n1 + n2);

  typedef csr_matrix::column_indices_array_type::view column_indices_view;
  typedef csr_matrix::row_offsets_array_type::view row_offsets_view;
  typedef csr_matrix::values_array_type::view values_view;

  column_indices_view jacE_column_indices(
    jacE_.column_indices.begin(),
    jacE_.column_indices.begin() + n1
    );


  row_offsets_view jacE_row_offsets(
    jacE_.row_offsets.begin(),
    jacE_.row_offsets.begin() + numPoints + 1
    );

  values_view jacE_values(
    jacE_.values.begin(),
    jacE_.values.begin() + n1
    );

  std::cout << "(0)" << std::endl;

  {
    cusp::array1d<int, cusp::host_memory> column_indices(jacE_column_indices.size());

    for (int i = 0; i < n1; ++i)
    {
      //column_indices[i] = i;

      if (i % numParams < numDimensions)
      {
        column_indices[i] = i % numDimensions + numDimensions * (i / numParams);
      }
      else
      {
        column_indices[i] = i % numDimensions + numDimensions * (i / numParams) + numDimensions * numPoints;
      }
    }

    std::cout << "(a)" << std::endl;

    cusp::copy(column_indices, jacE_column_indices);

    std::cout << "(b)" << std::endl;

    cusp::array1d<int, cusp::host_memory> row_offsets(jacE_row_offsets.size());

    row_offsets[0] = 0;
    for (int i = 1; i <= numPoints; ++i)
    {
      row_offsets[i] = row_offsets[i - 1] + numParams;
    }

    std::cout << "(c)" << std::endl;

    cusp::copy(row_offsets, jacE_row_offsets);

    std::cout << "(d)" << std::endl;
  }

  std::cout << "(1)" << std::endl;

  array1d_device_view ei(
    e_.begin() + numPoints,
    e_.begin() + numPoints + numPairs
    );

  column_indices_view jacEi_column_indices(
    jacE_.column_indices.begin() + n1,
    jacE_.column_indices.begin() + n1 + n2
    );

  row_offsets_view jacEi_row_offsets(
    jacE_.row_offsets.begin() + numPoints,
    jacE_.row_offsets.begin() + numPoints + numPairs + 1
    );

  values_view jacEi_values(
    jacE_.values.begin() + n1,
    jacE_.values.begin() + n1 + n2
    );

  {
    cusp::array1d<int, cusp::host_memory> column_indices(jacEi_column_indices.size());

    for (int i = 0; i < n2; ++i)
    {
      int k;
      if (i % numParamsPlusNumParams < numParams)
      {
        k = (i % numParams) + numParams * pIndPi[i / numParamsPlusNumParams];
      }
      else
      {
        k = (i % numParams) + numParams * pIndPj[i / numParamsPlusNumParams];
      }

      if (k % numParams < numDimensions)
      {
        column_indices[i] = k % numDimensions + numDimensions * (k / numParams);
      }
      else
      {
        column_indices[i] = k % numDimensions + numDimensions * (k / numParams) + numDimensions * numPoints;
      }
    }

    std::cout << "(a)" << std::endl;

    cusp::copy(column_indices, jacEi_column_indices);

    std::cout << "(b)" << std::endl;

    cusp::array1d<int, cusp::host_memory> row_offsets(jacEi_row_offsets.size());

    row_offsets[0] = jacEi_row_offsets[0];
    for (int i = 1; i <= numPairs; ++i)
    {
      row_offsets[i] = row_offsets[i - 1] + numParamsPlusNumParams;
    }

    std::cout << "(c)" << std::endl;

    cusp::copy(row_offsets, jacEi_row_offsets);

    std::cout << "(d)" << std::endl;
  }

  std::cout << "(2)" << std::endl;

  std::cout << "(3)" << std::endl;

  float unaryCostFunction1, unaryCostFunction2;
  float pairwiseCostFunction1, pairwiseCostFunction2;

  const float E = 2;
  const float D = 1 / E;

  typedef cusp::array2d<float, cusp::device_memory>::values_array_type::iterator iterator;
  cusp::array2d<float, cusp::device_memory> sAndT(numPoints, numParams);
  cusp::array2d_view<cusp::array1d_view<iterator> > s(numPoints, numDimensions, numDimensions,
    cusp::array1d_view<iterator>(sAndT.values.begin(), sAndT.values.begin() + numPoints * numDimensions)
    );

  cusp::array2d_view<cusp::array1d_view<iterator> > t(numPoints, numDimensions, numDimensions,
    cusp::array1d_view<iterator>(sAndT.values.begin() + numPoints * numDimensions, sAndT.values.end())
    );

  cusp::array2d<float, cusp::device_memory> sAndTPlusX(numPoints, numParams);

  cusp::array2d_view<cusp::array1d_view<iterator> > sPlusX(numPoints, numDimensions, numDimensions,
    cusp::array1d_view<iterator>(sAndTPlusX.values.begin(), sAndTPlusX.values.begin() + numPoints * numDimensions)
    );

  cusp::array2d_view<cusp::array1d_view<iterator> > tPlusX(numPoints, numDimensions, numDimensions,
    cusp::array1d_view<iterator>(sAndTPlusX.values.begin() + numPoints * numDimensions, sAndTPlusX.values.end())
    );

  cusp::copy(hS, s);
  cusp::copy(hT, t);
  std::cout << "(5)" << std::endl;

  UnaryCostGradientWithRespectToParams<float, numDimensions>(
    thrust::raw_pointer_cast(&tildeP(0, 0)),
    thrust::raw_pointer_cast(&s(0, 0)),
    thrust::raw_pointer_cast(&t(0, 0)),
    thrust::raw_pointer_cast(&beta[0]),
    thrust::raw_pointer_cast(&e[0]),
    thrust::raw_pointer_cast(&jacE_.values[0]),
    numPoints
    );

  std::cout << "(6)" << std::endl;

  PairwiseCostGradientWithRespectToParams<float, int, numDimensions>(
    thrust::raw_pointer_cast(&tildeP(0, 0)),
    thrust::raw_pointer_cast(&s(0, 0)),
    thrust::raw_pointer_cast(&t(0, 0)),
    thrust::raw_pointer_cast(&indPi[0]),
    thrust::raw_pointer_cast(&indPj[0]),
    thrust::raw_pointer_cast(&gamma[0]),
    thrust::raw_pointer_cast(&ei[0]),
    thrust::raw_pointer_cast(&jacE_.values[jacEi_row_offsets[0]]),
    numPairs
    );

  int iter = 0;
  float damp = 1;
  const float minDamp = 1e-5;

  std::cout << "(7)" << std::endl;
  while (iter < maxIterations)
  {
    std::cout << "Unary cost function " << (unaryCostFunction1 = cusp::blas::dot(e, e)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction1 = cusp::blas::dot(ei, ei)) << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    csr_matrix const& A = jacE_;
    cusp::array1d<float, cusp::device_memory> const& b = e_;

    float atol = 1e-6;
    float btol = 1e-6;
    float conlim = 0;

    int itnlim = 5000;

    csr_matrix At;
    cusp::transpose(A, At);

    cusp::array1d<float, cusp::device_memory> g(A.num_cols);
    cusp::multiply(At, b, g);

    cusp::array1d<float, cusp::device_memory> x(A.num_cols);

    SparseLeastSquares<csr_matrix> leastsq(A, At, b);
    leastsq.Solve(x, atol, btol, conlim, damp, itnlim, true);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cusp::blas::axpby(sAndT.values, x, sAndTPlusX.values, 1, -1);

    UnaryCostFunction<float, numDimensions>(
      thrust::raw_pointer_cast(&tildeP(0, 0)),
      thrust::raw_pointer_cast(&sPlusX(0, 0)),
      thrust::raw_pointer_cast(&tPlusX(0, 0)),
      thrust::raw_pointer_cast(&beta[0]),
      thrust::raw_pointer_cast(&edBar[0]),
      numPoints
      );

    PairwiseCostFunction<float, int, numDimensions>(
      thrust::raw_pointer_cast(&tildeP(0, 0)),
      thrust::raw_pointer_cast(&sPlusX(0, 0)),
      thrust::raw_pointer_cast(&tPlusX(0, 0)),
      thrust::raw_pointer_cast(&indPi[0]),
      thrust::raw_pointer_cast(&indPj[0]),
      thrust::raw_pointer_cast(&gamma[0]),
      thrust::raw_pointer_cast(&esBar[0]),
      numPairs
      );

    std::cout << "Unary cost function " << (unaryCostFunction2 = cusp::blas::dot(edBar, edBar)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction2 = cusp::blas::dot(esBar, esBar)) << std::endl;

    float FAtX = unaryCostFunction1 + pairwiseCostFunction1;
    float FAtXPlusY = unaryCostFunction2 + pairwiseCostFunction2;

    float numRho = FAtX - FAtXPlusY;
    float denRho = FAtX - leastsq.r1norm * leastsq.r1norm;

    float rho = numRho / denRho;

    std::cout << "num(rho) " << numRho << std::endl;
    std::cout << "den(rho) " << denRho << std::endl;

    if (rho != rho)
    {
      std::cout << "rho is nan" << std::endl;
      break;
    }
    else if (rho < 0.01)
    {
      std::cout << "rho < pi1" << std::endl;
      if (damp == 0)
      {
        damp = minDamp;
        std::cout << "damp = minDamp" << std::endl;
      }
      else
      {
        damp = E * damp;
        std::cout << "damp = E * damp" << std::endl;
      }
    }
    else
    {
      bool convergence = false;

      const float tolx = 1e-6;
      const float tolf = 1e-6;
      const float tolg = 1e-5;

      float ynorm = cusp::blas::nrmmax(x);
      float xnorm = cusp::blas::nrmmax(sAndT.values);
      float xPlusYnorm = cusp::blas::nrmmax(sAndTPlusX.values);

      std::cout << "ynorm " << ynorm << std::endl;
      std::cout << "xPlusYnorm " << xPlusYnorm << std::endl;
      std::cout << "xnorm " << xnorm << std::endl;

      //if (ynorm / (xPlusYnorm + xnorm) <= tolx)
      if (ynorm <= tolx)
      {
        convergence = true;
        std::cout << "x-convergence criterion is signalled" << std::endl;
      }

      if ((FAtX - FAtXPlusY) / FAtXPlusY <= tolf)
      {
        convergence = true;
        std::cout << "Function convergence criterion is signalled" << std::endl;
      }

      float gnorm = cusp::blas::nrm2(g);
      std::cout << "gnorm " << gnorm << std::endl;

      if (gnorm <= tolg)
      {
        convergence = true;
        std::cout << "Gradient convergence criterion is signalled" << std::endl;
      }

      if (convergence) break;

      cusp::copy(sAndTPlusX, sAndT);

      std::cout << "(5)" << std::endl;

      UnaryCostGradientWithRespectToParams<float, numDimensions>(
        thrust::raw_pointer_cast(&tildeP(0, 0)),
        thrust::raw_pointer_cast(&s(0, 0)),
        thrust::raw_pointer_cast(&t(0, 0)),
        thrust::raw_pointer_cast(&beta[0]),
        thrust::raw_pointer_cast(&e[0]),
        thrust::raw_pointer_cast(&jacE_.values[0]),
        numPoints
        );

      std::cout << "(6)" << std::endl;

      PairwiseCostGradientWithRespectToParams<float, int, numDimensions>(
        thrust::raw_pointer_cast(&tildeP(0, 0)),
        thrust::raw_pointer_cast(&s(0, 0)),
        thrust::raw_pointer_cast(&t(0, 0)),
        thrust::raw_pointer_cast(&indPi[0]),
        thrust::raw_pointer_cast(&indPj[0]),
        thrust::raw_pointer_cast(&gamma[0]),
        thrust::raw_pointer_cast(&ei[0]),
        thrust::raw_pointer_cast(&jacE_.values[jacEi_row_offsets[0]]),
        numPairs
        );

      std::cout << "(7)" << std::endl;

      iter = iter + 1;

      if (rho > 0.75)
      {
        std::cout << "rho > pi2" << std::endl;
        damp = D * damp;
        std::cout << "damp = D damp" << std::endl;
      }
      else
      {
        std::cout << "pi1 < rho < pi2" << std::endl;
      }

      if (damp < minDamp)
      {
        damp = 0;
        std::cout << "damp <- 0" << std::endl;
      }
    }

    if (damp > 32000)
    {
      std::cout << "damp > maxDamp " << std::endl;
      break;
    }

    std::cout << "rho " << rho << std::endl;
    std::cout << "damp " << damp << std::endl;
    std::cout << "iter " << iter << std::endl;
  }

  ProjectionOntoLine<float, 3>(
    thrust::raw_pointer_cast(&tildeP(0, 0)),
    thrust::raw_pointer_cast(&s(0, 0)),
    thrust::raw_pointer_cast(&t(0, 0)),
    thrust::raw_pointer_cast(&p(0, 0)),
    numPoints
    );

  cusp::copy(s, hS);
  cusp::copy(t, hT);
  cusp::copy(p, hP);
}
