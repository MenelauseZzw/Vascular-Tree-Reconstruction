#include "TestLevenbergMarquardtMinimizer.h"
#include "PairwiseCostFunctionAndItsGradientWithRespectToParams.h"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/blas.h>
#include <cusp/csr_matrix.h>
#include <cusp/copy.h>
#include <cusp/ell_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/krylov/cg_m.h>
#include <cusp/linear_operator.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>
#include <iostream>
#include <thrust/device_ptr.h>

void testLevenbergMarquardtMinimizer(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs)
{
  int numDims = 3;

  typedef cusp::array1d_view<float*> array1d_view;
  typedef cusp::array2d_view<array1d_view> array2d_view;

  array2d_view hTildeP(numPoints, numDims, numDims, array1d_view(pTildeP, pTildeP + numDims * numPoints));
  array2d_view hS(numPoints, numDims, numDims, array1d_view(pS, pS + numDims * numPoints));
  array2d_view hT(numPoints, numDims, numDims, array1d_view(pT, pT + numDims * numPoints));
  array1d_view hSigma(pSigma, pSigma + numPoints);

  const int numParams = 6;
  const int numParamsPlusNumParams = numParams + numParams;

  typedef cusp::array2d<float, cusp::host_memory> array2d;

  array2d hJacTildeP(numDims * numPoints, numParams, 0);
  array2d hJacS(numDims * numPoints, numParams, 0);
  array2d hJacT(numDims * numPoints, numParams, 0);

  for (int i = 0; i < numDims * numPoints; i += 3)
  {
    hJacS(i + 0, 0) = 1;
    hJacS(i + 1, 1) = 1;
    hJacS(i + 2, 2) = 1;

    hJacT(i + 0, 3) = 1;
    hJacT(i + 1, 4) = 1;
    hJacT(i + 2, 5) = 1;
  }

  array2d hJacTildePi(numDims * numPoints, numParams + numParams, 0);
  array2d hJacSi(numDims * numPoints, numParams + numParams, 0);
  array2d hJacTi(numDims * numPoints, numParams + numParams, 0);

  for (int i = 0; i < numDims * numPoints; i += 3)
  {
    hJacSi(i + 0, 0) = 1;
    hJacSi(i + 1, 1) = 1;
    hJacSi(i + 2, 2) = 1;

    hJacTi(i + 0, 3) = 1;
    hJacTi(i + 1, 4) = 1;
    hJacTi(i + 2, 5) = 1;
  }

  array2d hJacTildePj(numDims * numPoints, numParams + numParams, 0);
  array2d hJacSj(numDims * numPoints, numParams + numParams, 0);
  array2d hJacTj(numDims * numPoints, numParams + numParams, 0);

  for (int i = 0; i < numDims * numPoints; i += 3)
  {
    hJacSj(i + 0, 6) = 1;
    hJacSj(i + 1, 7) = 1;
    hJacSj(i + 2, 8) = 1;

    hJacTj(i + 0, 9) = 1;
    hJacTj(i + 1, 10) = 1;
    hJacTj(i + 2, 11) = 1;
  }

  cusp::ell_matrix<int, int, cusp::host_memory> hIndPi(numPairs, numPoints, numPairs, 1);
  cusp::ell_matrix<int, int, cusp::host_memory> hIndPj(numPairs, numPoints, numPairs, 1);

  for (int i = 0; i < numPairs; ++i)
  {
    hIndPi.column_indices(i, 0) = pIndPi[i];
    hIndPi.values(i, 0) = 1;

    hIndPj.column_indices(i, 0) = pIndPj[i];
    hIndPj.values(i, 0) = 1;
  }

  cusp::array2d<float, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<float, cusp::device_memory> sigma(hSigma);

  cusp::array2d<float, cusp::device_memory> jacTildeP(hJacTildeP);
  cusp::array2d<float, cusp::device_memory> jacS(hJacS);
  cusp::array2d<float, cusp::device_memory> jacT(hJacT);

  cusp::array2d<float, cusp::host_memory> hTildePi(numPairs, numDims);
  cusp::array2d<float, cusp::host_memory> hSi(numPairs, numDims);
  cusp::array2d<float, cusp::host_memory> hTi(numPairs, numDims);

  cusp::array2d<float, cusp::host_memory> hTildePj(numPairs, numDims);
  cusp::array2d<float, cusp::host_memory> hSj(numPairs, numDims);
  cusp::array2d<float, cusp::host_memory> hTj(numPairs, numDims);

  cusp::array2d<float, cusp::device_memory> jacTildePi(hJacTildePi);
  cusp::array2d<float, cusp::device_memory> jacSi(hJacSi);
  cusp::array2d<float, cusp::device_memory> jacTi(hJacTi);

  cusp::array2d<float, cusp::device_memory> jacTildePj(hJacTildePj);
  cusp::array2d<float, cusp::device_memory> jacSj(hJacSj);
  cusp::array2d<float, cusp::device_memory> jacTj(hJacTj);

  cusp::ell_matrix<int, int, cusp::device_memory> indPi(hIndPi);
  cusp::ell_matrix<int, int, cusp::device_memory> indPj(hIndPj);

  cusp::array1d<float, cusp::device_memory> e(numPoints);

  typedef cusp::csr_matrix<int, float, cusp::device_memory> csr_matrix;

  csr_matrix jacE(numPoints, numPoints * numParams, numPoints * numParams);
  {
    cusp::array1d<float, cusp::host_memory> column_indices(jacE.num_cols);
    for (int i = 0; i < jacE.num_entries; ++i)
    {
      column_indices[i] = i;
    }
    jacE.column_indices = column_indices;

    cusp::array1d<float, cusp::host_memory> row_offsets(jacE.num_rows + 1);
    for (int i = 0; i <= jacE.num_rows; ++i)
    {
      row_offsets[i] = i * numParams;
    }
    jacE.row_offsets = row_offsets;
  }

  cusp::array1d<float, cusp::device_memory> ei(numPairs);
  cusp::array1d<float, cusp::device_memory> ej(numPairs);

  csr_matrix jacEi(numPairs, numPoints * numParams, numPairs * numParamsPlusNumParams);
  {
    cusp::array1d<float, cusp::host_memory> column_indices(jacEi.num_entries);
    for (int i = 0; i < jacEi.num_entries; ++i)
    {
      if (i % numParamsPlusNumParams < numParams)
      {
        column_indices[i] = (i % numParams) + numParams * pIndPi[i / numParamsPlusNumParams];
      }
      else
      {
        column_indices[i] = (i % numParams) + numParams * pIndPj[i / numParamsPlusNumParams];
      }
    }

    jacEi.column_indices = column_indices;

    cusp::array1d<float, cusp::host_memory> row_offsets(jacEi.num_rows + 1);
    for (int i = 0; i <= jacEi.num_rows; ++i)
    {
      row_offsets[i] = i * numParamsPlusNumParams;
    }
    jacEi.row_offsets = row_offsets;
  }

  csr_matrix jacEj(numPairs, numPoints * numParams, numPairs * numParamsPlusNumParams);
  {
    cusp::array1d<float, cusp::host_memory> column_indices(jacEj.num_entries);
    for (int i = 0; i < jacEj.num_entries; ++i)
    {
      if (i % numParamsPlusNumParams < numParams)
      {
        column_indices[i] = (i % numParams) + numParams * pIndPi[i / numParamsPlusNumParams];
      }
      else
      {
        column_indices[i] = (i % numParams) + numParams * pIndPj[i / numParamsPlusNumParams];
      }
    }
    jacEj.column_indices = column_indices;

    cusp::array1d<float, cusp::host_memory> row_offsets(jacEj.num_rows + 1);
    for (int i = 0; i <= jacEj.num_rows; ++i)
    {
      row_offsets[i] = i * numParamsPlusNumParams;
    }
    jacEj.row_offsets = row_offsets;
  }

  float unaryCostFunction1, unaryCostFunction2;
  float pairwiseCostFunction1, pairwiseCostFunction2;

  const float E = 2;
  const float D = 1 / E;

  cusp::array1d<float, cusp::device_memory> lambda(1, 100);
  const int maxNumPoints = 65535;

  for (int iter = 0; iter < 10; ++iter)
  {
    cusp::array2d<float, cusp::device_memory> s(hS);
    cusp::array2d<float, cusp::device_memory> t(hT);

    for (int numPnt0 = 0; numPnt0 < numPoints; numPnt0 += maxNumPoints)
    {
      UnaryCostFunctionAndItsGradientWithRespectToParams3x6(
        thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&s(numPnt0, 0)),
        thrust::raw_pointer_cast(&t(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacS(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacT(numPnt0, 0)),
        thrust::raw_pointer_cast(&e[numPnt0]),
        thrust::raw_pointer_cast(&jacE.values[jacE.row_offsets[numPnt0]]),
        std::min(numPoints - numPnt0, maxNumPoints)
        );
    }

    csr_matrix jacEt;
    cusp::array1d<float, cusp::device_memory> jacEtTimesE(numPoints * numParams);
    csr_matrix jacEtTimesJacE;

    cusp::transpose(jacE, jacEt);
    cusp::multiply(jacEt, e, jacEtTimesE);
    cusp::multiply(jacEt, jacE, jacEtTimesJacE);

    cusp::multiply(hIndPi, hTildeP, hTildePi);
    cusp::multiply(hIndPi, hS, hSi);
    cusp::multiply(hIndPi, hT, hTi);

    cusp::multiply(hIndPj, hTildeP, hTildePj);
    cusp::multiply(hIndPj, hS, hSj);
    cusp::multiply(hIndPj, hT, hTj);

    cusp::array2d<float, cusp::device_memory> tildePi(hTildePi);
    cusp::array2d<float, cusp::device_memory> si(hSi);
    cusp::array2d<float, cusp::device_memory> ti(hTi);

    cusp::array2d<float, cusp::device_memory> tildePj(hTildePj);
    cusp::array2d<float, cusp::device_memory> sj(hSj);
    cusp::array2d<float, cusp::device_memory> tj(hTj);

    for (int numPnt0 = 0; numPnt0 < numPairs; numPnt0 += maxNumPoints)
    {
      PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(
        thrust::raw_pointer_cast(&tildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&si(numPnt0, 0)),
        thrust::raw_pointer_cast(&ti(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTi(numPnt0, 0)),
        thrust::raw_pointer_cast(&tildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&sj(numPnt0, 0)),
        thrust::raw_pointer_cast(&tj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTj(numPnt0, 0)),
        thrust::raw_pointer_cast(&ei[numPnt0]),
        thrust::raw_pointer_cast(&ej[numPnt0]),
        thrust::raw_pointer_cast(&jacEi.values[jacEi.row_offsets[numPnt0]]),
        thrust::raw_pointer_cast(&jacEj.values[jacEj.row_offsets[numPnt0]]),
        std::min(numPairs - numPnt0, maxNumPoints)
        );
    }

    {
      csr_matrix jacEit;
      cusp::array1d<float, cusp::device_memory> jacEitTimesEi(numPoints * numParams);
      csr_matrix jacEitTimesJacEi;

      cusp::transpose(jacEi, jacEit);
      cusp::multiply(jacEit, ei, jacEitTimesEi);
      cusp::multiply(jacEit, jacEi, jacEitTimesJacEi);

      cusp::blas::scal(jacEtTimesE, -1);
      cusp::blas::axpy(jacEitTimesEi, jacEtTimesE, -1);
      cusp::add(jacEtTimesJacE, jacEitTimesJacEi, jacEtTimesJacE);
    }

    {
      csr_matrix jacEjt;
      cusp::array1d<float, cusp::device_memory> jacEjtTimesEj(numPoints * numParams);
      csr_matrix jacEjtTimesJacEj;

      cusp::transpose(jacEj, jacEjt);
      cusp::multiply(jacEjt, ej, jacEjtTimesEj);
      cusp::multiply(jacEjt, jacEj, jacEjtTimesJacEj);

      cusp::blas::axpy(jacEjtTimesEj, jacEtTimesE, -1);
      cusp::add(jacEtTimesJacE, jacEjtTimesJacEj, jacEtTimesJacE);
    }

    std::cout << "Unary cost function " << (unaryCostFunction1 = cusp::blas::dot(e, e)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction1 = cusp::blas::dot(ei, ei) + cusp::blas::dot(ej, ej)) << std::endl;

    cusp::convergence_monitor<float> monitor(jacEtTimesE, 1000, 1e-6);

    cusp::array1d<float, cusp::device_memory> x(numPoints * numParams, 0);

    cusp::krylov::cg_m(jacEtTimesJacE, x, jacEtTimesE, lambda, monitor);

    if (monitor.converged())
    {
      std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
      std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl;
    }
    else
    {
      std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
      std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
    }

    cusp::array1d<float, cusp::host_memory> hX(x);
    cusp::array2d<float, cusp::host_memory> hSPlusX(hS);
    cusp::array2d<float, cusp::host_memory> hTPlusX(hT);

    for (int i = 0; i < x.size(); ++i)
    {
      int ind = i % numDims;
      if ((i % numParams) < numParams / 2)
      {
        hSPlusX(i / numParams, ind) += hX[i];
      }
      else
      {
        hTPlusX(i / numParams, ind) += hX[i];
      }
    }

    cusp::array2d<float, cusp::device_memory> sPlusX(hSPlusX);
    cusp::array2d<float, cusp::device_memory> tPlusX(hTPlusX);
    
    for (int numPnt0 = 0; numPnt0 < numPoints; numPnt0 += maxNumPoints)
    {
      UnaryCostFunctionAndItsGradientWithRespectToParams3x6(
        thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&sPlusX(numPnt0, 0)),
        thrust::raw_pointer_cast(&tPlusX(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacS(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacT(numPnt0, 0)),
        thrust::raw_pointer_cast(&e[numPnt0]),
        thrust::raw_pointer_cast(&jacE.values[jacE.row_offsets[numPnt0]]),
        std::min(numPoints - numPnt0, maxNumPoints)
        );
    }

    cusp::array2d<float, cusp::host_memory> hSPlusXi(numPairs, numDims);
    cusp::array2d<float, cusp::host_memory> hTPlusXi(numPairs, numDims);

    cusp::array2d<float, cusp::host_memory> hSPlusXj(numPairs, numDims);
    cusp::array2d<float, cusp::host_memory> hTPlusXj(numPairs, numDims);

    cusp::multiply(hIndPi, hSPlusX, hSPlusXi);
    cusp::multiply(hIndPi, hTPlusX, hTPlusXi);

    cusp::multiply(hIndPj, hSPlusX, hSPlusXj);
    cusp::multiply(hIndPj, hTPlusX, hTPlusXj);

    cusp::array2d<float, cusp::device_memory> sPlusXi(hSPlusXi);
    cusp::array2d<float, cusp::device_memory> tPlusXi(hTPlusXi);

    cusp::array2d<float, cusp::device_memory> sPlusXj(hSPlusXj);
    cusp::array2d<float, cusp::device_memory> tPlusXj(hTPlusXj);

    for (int numPnt0 = 0; numPnt0 < numPairs; numPnt0 += maxNumPoints)
    {
      PairwiseCostFunctionAndItsGradientWithRespectToParams3x12(
        thrust::raw_pointer_cast(&tildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&sPlusXi(numPnt0, 0)),
        thrust::raw_pointer_cast(&tPlusXi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTi(numPnt0, 0)),
        thrust::raw_pointer_cast(&tildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&sPlusXj(numPnt0, 0)),
        thrust::raw_pointer_cast(&tPlusXj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTj(numPnt0, 0)),
        thrust::raw_pointer_cast(&ei[numPnt0]),
        thrust::raw_pointer_cast(&ej[numPnt0]),
        thrust::raw_pointer_cast(&jacEi.values[jacEi.row_offsets[numPnt0]]),
        thrust::raw_pointer_cast(&jacEj.values[jacEj.row_offsets[numPnt0]]),
        std::min(numPairs - numPnt0, maxNumPoints)
        );
    }

    std::cout << "Unary cost function " << (unaryCostFunction2 = cusp::blas::dot(e, e)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction2 = cusp::blas::dot(ei, ei) + cusp::blas::dot(ej, ej)) << std::endl;

    if (unaryCostFunction1 + pairwiseCostFunction1 > unaryCostFunction2 + pairwiseCostFunction2)
    {
      cusp::copy(hSPlusX.values, hS.values);
      cusp::copy(hTPlusX.values, hT.values);
      lambda[0] *= D;

      std::cout << "lambda = D lambda" << std::endl;
    }
    else
    {
      lambda[0] *= E;

      std::cout << "lambda = E lambda" << std::endl;
    }
    std::cout << "lambda " << lambda[0] << std::endl;
  }
}
