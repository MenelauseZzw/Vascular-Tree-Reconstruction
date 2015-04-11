#include "TestLevenbergMarquardtMinimizer1.h"
#include "AdjustLineEndpoints.h"
#include "PairwiseCostFunctionAndItsGradientWithRespectToParams.h"
#include "ProjectionOntoLineAndItsJacobian.h"
#include "UnaryCostFunctionAndItsGradientWithRespectToParams.h"
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

void testLevenbergMarquardtMinimizer1(float* pTildeP, float* pS, float* pT, float* pSigma, int numPoints, int* pIndPi, int* pIndPj, int numPairs, float* pP)
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

  cusp::array1d_view<int*> hIndPi(pIndPi, pIndPi + numPairs);
  cusp::array1d_view<int*> hIndPj(pIndPj, pIndPj + numPairs);

  cusp::array2d<float, cusp::device_memory> tildeP(hTildeP);
  cusp::array1d<float, cusp::device_memory> sigma(hSigma);

  cusp::array2d<float, cusp::device_memory> jacTildeP(hJacTildeP);
  cusp::array2d<float, cusp::device_memory> jacS(hJacS);
  cusp::array2d<float, cusp::device_memory> jacT(hJacT);

  cusp::array2d<float, cusp::device_memory> tildePi(numPairs, numDims);
  cusp::array2d<float, cusp::device_memory> si(numPairs, numDims);
  cusp::array2d<float, cusp::device_memory> ti(numPairs, numDims);

  cusp::array2d<float, cusp::device_memory> tildePj(numPairs, numDims);
  cusp::array2d<float, cusp::device_memory> sj(numPairs, numDims);
  cusp::array2d<float, cusp::device_memory> tj(numPairs, numDims);

  cusp::array2d<float, cusp::device_memory> jacTildePi(hJacTildePi);
  cusp::array2d<float, cusp::device_memory> jacSi(hJacSi);
  cusp::array2d<float, cusp::device_memory> jacTi(hJacTi);

  cusp::array2d<float, cusp::device_memory> jacTildePj(hJacTildePj);
  cusp::array2d<float, cusp::device_memory> jacSj(hJacSj);
  cusp::array2d<float, cusp::device_memory> jacTj(hJacTj);

  cusp::array1d<int, cusp::device_memory> indPi(hIndPi);
  cusp::array1d<int, cusp::device_memory> indPj(hIndPj);

  cusp::array1d<float, cusp::device_memory> e_(numPoints + numPairs + numPairs, 0);

  typedef cusp::array1d<float, cusp::device_memory>::view array1d_device_view;
  array1d_device_view e(
    e_.begin(),
    e_.begin() + numPoints
    );

  typedef cusp::csr_matrix<int, float, cusp::device_memory> csr_matrix;

  const int n1 = numPoints * numParams;
  const int n2 = numPairs * numParamsPlusNumParams;
  const int n3 = numPairs * numParamsPlusNumParams;

  csr_matrix jacE_(numPoints + numPairs + numPairs, numPoints * numParams, n1 + n2 + n3);

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

      if (i % numParams < numDims)
      {
        column_indices[i] = i % numDims + numDims * (i / numParams);
      }
      else
      {
        column_indices[i] = i % numDims + numDims * (i / numParams) + numDims * numPoints;
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

  array1d_device_view ej(
    e_.begin() + numPoints + numPairs,
    e_.begin() + numPoints + numPairs + numPairs
    );

  //csr_matrix jacEi(numPairs, numPoints * numParams, numPairs * numParamsPlusNumParams);

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

      if (k % numParams < numDims)
      {
        column_indices[i] = k % numDims + numDims * (k / numParams);
      }
      else
      {
        column_indices[i] = k % numDims + numDims * (k / numParams) + numDims * numPoints;
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

  //csr_matrix jacEj(numPairs, numPoints * numParams, numPairs * numParamsPlusNumParams);

  column_indices_view jacEj_column_indices(
    jacE_.column_indices.begin() + n1 + n2,
    jacE_.column_indices.begin() + n1 + n2 + n3
    );

  row_offsets_view jacEj_row_offsets(
    jacE_.row_offsets.begin() + numPoints + numPairs,
    jacE_.row_offsets.begin() + numPoints + numPairs + numPairs + 1
    );

  values_view jacEj_values(
    jacE_.values.begin() + n1 + n2,
    jacE_.values.begin() + n1 + n2 + n3
    );

  {
    cusp::array1d<int, cusp::host_memory> column_indices(jacEj_column_indices.size());

    /*  for (int i = 0; i < n3; ++i)
      {
      if (i % numParamsPlusNumParams < numParams)
      {
      column_indices[i] = (i % numParams) + numParams * pIndPi[i / numParamsPlusNumParams];
      }
      else
      {
      column_indices[i] = (i % numParams) + numParams * pIndPj[i / numParamsPlusNumParams];
      }
      }*/

    for (int i = 0; i < n3; ++i)
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

      if (k % numParams < numDims)
      {
        column_indices[i] = k % numDims + numDims * (k / numParams);
      }
      else
      {
        column_indices[i] = k % numDims + numDims * (k / numParams) + numDims * numPoints;
      }
    }

    std::cout << "(a)" << std::endl;

    cusp::copy(column_indices, jacEj_column_indices);

    std::cout << "(b)" << std::endl;

    cusp::array1d<int, cusp::host_memory> row_offsets(jacEj_row_offsets.size());

    row_offsets[0] = jacEj_row_offsets[0];

    for (int i = 1; i <= numPairs; ++i)
    {
      row_offsets[i] = row_offsets[i - 1] + numParamsPlusNumParams;
    }

    std::cout << "(c)" << std::endl;

    cusp::copy(row_offsets, jacEj_row_offsets);

    std::cout << "(d)" << std::endl;
  }

  std::cout << "(3)" << std::endl;

  float unaryCostFunction1, unaryCostFunction2;
  float pairwiseCostFunction1, pairwiseCostFunction2;

  const float E = 2;
  const float D = 1 / E;

  float lambda = 100;
  const int maxNumPoints = 65535;

  typedef cusp::array2d<float, cusp::device_memory>::values_array_type::iterator iterator;
  cusp::array2d<float, cusp::device_memory> sAndT(numPoints, numParams);
  cusp::array2d_view<cusp::array1d_view<iterator> > s(numPoints, numDims, numDims,
    cusp::array1d_view<iterator>(sAndT.values.begin(), sAndT.values.begin() + numPoints * numDims)
    );

  cusp::array2d_view<cusp::array1d_view<iterator> > t(numPoints, numDims, numDims,
    cusp::array1d_view<iterator>(sAndT.values.begin() + numPoints * numDims, sAndT.values.end())
    );

  cusp::array2d<float, cusp::device_memory> sAndTPlusX(numPoints, numParams);

  cusp::array2d_view<cusp::array1d_view<iterator> > sPlusX(numPoints, numDims, numDims,
    cusp::array1d_view<iterator>(sAndTPlusX.values.begin(), sAndTPlusX.values.begin() + numPoints * numDims)
    );

  cusp::array2d_view<cusp::array1d_view<iterator> > tPlusX(numPoints, numDims, numDims,
    cusp::array1d_view<iterator>(sAndTPlusX.values.begin() + numPoints * numDims, sAndTPlusX.values.end())
    );

  //cusp::array2d<float, cusp::device_memory> s(hS);
  //cusp::array2d<float, cusp::device_memory> t(hT);


  cusp::copy(hS, s);
  cusp::copy(hT, t);

  for (int iter = 0; iter < 200; ++iter)
  {
    std::cout << "(4)" << std::endl;

    for (int numPnt0 = 0; numPnt0 < numPoints; numPnt0 += maxNumPoints)
    {
      UnaryCostFunctionAndItsGradientWithRespectToParams3x6(
        thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&s(numPnt0, 0)),
        thrust::raw_pointer_cast(&t(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacS(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacT(numPnt0, 0)),
        thrust::raw_pointer_cast(&sigma[numPnt0]),
        thrust::raw_pointer_cast(&e[numPnt0]),
        thrust::raw_pointer_cast(&jacE_.values[jacE_row_offsets[numPnt0]]),
        std::min(numPoints - numPnt0, maxNumPoints)
        );

      std::cout << "(5)" << std::endl;
    }

    std::cout << "(6)" << std::endl;

    for (int numPnt0 = 0; numPnt0 < numPairs; numPnt0 += maxNumPoints)
    {
      PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations3x12(
        thrust::raw_pointer_cast(&tildeP(0, 0)),
        thrust::raw_pointer_cast(&s(0, 0)),
        thrust::raw_pointer_cast(&t(0, 0)),
        thrust::raw_pointer_cast(&indPi[numPnt0]),
        thrust::raw_pointer_cast(&indPj[numPnt0]),
        thrust::raw_pointer_cast(&jacTildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTj(numPnt0, 0)),
        thrust::raw_pointer_cast(&ei[numPnt0]),
        thrust::raw_pointer_cast(&ej[numPnt0]),
        thrust::raw_pointer_cast(&jacE_.values[jacEi_row_offsets[numPnt0]]),
        thrust::raw_pointer_cast(&jacE_.values[jacEj_row_offsets[numPnt0]]),
        std::min(numPairs - numPnt0, maxNumPoints)
        );

      std::cout << "(7)" << std::endl;
    }

    std::cout << "Unary cost function " << (unaryCostFunction1 = cusp::blas::dot(e, e)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction1 = cusp::blas::dot(ei, ei) + cusp::blas::dot(ej, ej)) << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    csr_matrix const& A = jacE_;
    cusp::array1d<float, cusp::device_memory> const& b = e_;

    float atol = 1e-6;
    float btol = 1e-6;
    float conlim = 0;

    float damp = lambda;
    int itnlim = 5000;


    //if isa(A, 'numeric')
    //  explicitA = true;
    //elseif isa(A, 'function_handle')
    //  explicitA = false;
    //else
    //  error('SOL:lsqrSOL:Atype', '%s', 'A must be numeric or a function handle');
    //end

    //wantvar = nargout >= 10;
    //if wantvar, var = zeros(n, 1); end

    //msg = ['The exact solution is  x = 0                              '
    //'Ax - b is small enough, given atol, btol                  '
    //'The least-squares solution is good enough, given atol     '
    //'The estimate of cond(Abar) has exceeded conlim            '
    //'Ax - b is small enough for this machine                   '
    //'The least-squares solution is good enough for this machine'
    //'Cond(Abar) seems to be too large for this machine         '
    //'The iteration limit has been reached                      '];

    //if show
    //  disp(' ')
    //  disp('LSQR            Least-squares solution of  Ax = b')
    //  str1 = sprintf('The matrix A has %8g rows  and %8g cols', m, n);
    //  str2 = sprintf('damp = %20.14e    wantvar = %8g', damp, wantvar);
    //  str3 = sprintf('atol = %8.2e                 conlim = %8.2e', atol, conlim);
    //  str4 = sprintf('btol = %8.2e                 itnlim = %8g', btol, itnlim);
    //  disp(str1);   disp(str2);   disp(str3);   disp(str4);
    //end

    //itn = 0;             istop = 0;
    int itn = 0;
    int istop = 0;
    //ctol = 0;             if conlim > 0, ctol = 1 / conlim; end;
    float ctol = 0;
    if (conlim > 0)
    {
      ctol = 1 / conlim;
    }
    //Anorm = 0;             Acond = 0;
    float Anorm = 0;
    float Acond = 0;
    //dampsq = damp ^ 2;        ddnorm = 0;             res2 = 0;
    float dampsq = damp * damp;
    float ddnorm = 0;
    float res2 = 0;

    //xnorm = 0;             xxnorm = 0;             z = 0;
    float xnorm = 0;
    float xxnorm = 0;
    float z = 0;

    //cs2 = -1;            sn2 = 0;
    float cs2 = -1;
    float sn2 = 0;

    csr_matrix At;
    cusp::transpose(A, At);

    // Initialize.

    // Set up the first vectors u and v for the bidiagonalization.
    // These satisfy  beta*u = b, alfa*v = A'u.

    cusp::array1d<float, cusp::device_memory> u(A.num_rows);
    cusp::array1d<float, cusp::device_memory> v(A.num_cols);

    cusp::array1d<float, cusp::device_memory> Atu(A.num_cols);
    cusp::array1d<float, cusp::device_memory> Av(A.num_rows);

    cusp::array1d<float, cusp::device_memory> w(A.num_cols);
    cusp::array1d<float, cusp::device_memory> dk(A.num_cols);

    // u = b(1:m);        x = zeros(n, 1);
    cusp::copy(b, u);
    cusp::array1d<float, cusp::device_memory> x(A.num_cols, 0);

    //alfa = 0;             beta = norm(u);
    float alfa = 0;
    float beta = cusp::blas::nrm2(u);

    //if beta > 0
    if (beta > 0)
    {
      //u = (1 / beta)*u;
      cusp::blas::scal(u, 1 / beta);
      //if explicitA
      //  v = A'*u;
      //else
      //v = A(u, 2);
      //end
      cusp::multiply(At, u, v);
      //alfa = norm(v);
      alfa = cusp::blas::nrm2(v);
      //end  
    }

    //  if alfa > 0
    if (alfa > 0)
    {
      //    v = (1 / alfa)*v;      w = v;
      cusp::blas::scal(v, 1 / alfa);
      cusp::copy(v, w);
      //end
    }

    //  Arnorm = alfa*beta;     if Arnorm == 0, disp(msg(1, :)); return, end
    float Arnorm = alfa * beta;
    if (Arnorm == 0)
    {
      return;
    }

    //  rhobar = alfa;          phibar = beta;          bnorm = beta;
    float rhobar = alfa;
    float phibar = beta;
    float bnorm = beta;
    //rnorm = beta;
    float rnorm = beta;
    //r1norm = rnorm;
    float r1norm = rnorm;
    //r2norm = rnorm;
    float r2norm = rnorm;
    //head1 = '   Itn      x(1)       r1norm     r2norm ';
    //head2 = ' Compatible   LS      Norm A   Cond A';

    //if show
    //  disp(' ')
    //  disp([head1 head2])
    //  test1 = 1;          test2 = alfa / beta;
    //str1 = sprintf('%6g %12.5e', itn, x(1));
    //str2 = sprintf(' %10.3e %10.3e', r1norm, r2norm);
    //str3 = sprintf('  %8.1e %8.1e', test1, test2);
    //disp([str1 str2 str3])
    //  end

    //------------------------------------------------------------------
    //     Main iteration loop.
    //------------------------------------------------------------------
    //  while itn < itnlim
    //    itn = itn + 1;

    while (itn < itnlim)
    {
      itn = itn + 1;
      //% Perform the next step of the bidiagonalization to obtain the
      //  % next beta, u, alfa, v.These satisfy the relations
      //  %      beta*u = A*v - alfa*u,
      //  %      alfa*v = A'*u - beta*v.

      //  if explicitA
      //    u = A*v - alfa*u;
      //  else
      //    u = A(v, 1) - alfa*u;
      //end
      cusp::multiply(A, v, Av);
      cusp::blas::axpby(Av, u, u, 1, -alfa);

      //  beta = norm(u);
      beta = cusp::blas::nrm2(u);
      //if beta > 0
      if (beta > 0)
      {
        //  u = (1 / beta)*u;
        cusp::blas::scal(u, 1 / beta);
      }
      //Anorm = norm([Anorm alfa beta damp]);
      Anorm = sqrt(Anorm * Anorm + alfa * alfa + beta * beta + damp * damp);//?
      //if explicitA
      //  v = A'*u   - beta*v;
      //else
      //v = A(u, 2) - beta*v;
      //end
      cusp::multiply(At, u, Atu);
      cusp::blas::axpby(Atu, v, v, 1, -beta);

      //  alfa = norm(v);
      alfa = cusp::blas::nrm2(v);
      //if alfa > 0, v = (1 / alfa)*v; end
      //  end
      if (alfa > 0)
      {
        cusp::blas::scal(v, 1 / alfa);
      }

      //  % Use a plane rotation to eliminate the damping parameter.
      //  % This alters the diagonal(rhobar) of the lower - bidiagonal matrix.

      //rhobar1 = norm([rhobar damp]);
      //cs1 = rhobar / rhobar1;
      //sn1 = damp / rhobar1;
      //psi = sn1*phibar;
      //phibar = cs1*phibar;
      float rhobar1 = sqrt(rhobar * rhobar + damp * damp);
      float cs1 = rhobar / rhobar1;
      float sn1 = damp / rhobar1;
      float psi = sn1*phibar;
      phibar = cs1*phibar;

      //% Use a plane rotation to eliminate the subdiagonal element(beta)
      //  % of the lower - bidiagonal matrix, giving an upper - bidiagonal matrix.

      //  rho = norm([rhobar1 beta]);
      //cs = rhobar1 / rho;
      //sn = beta / rho;
      //theta = sn*alfa;
      //rhobar = -cs*alfa;
      //phi = cs*phibar;
      //phibar = sn*phibar;
      //tau = sn*phi;
      float rho = sqrt(rhobar1 * rhobar1 + beta * beta);
      float cs = rhobar1 / rho;
      float sn = beta / rho;
      float theta = sn*alfa;
      rhobar = -cs*alfa;
      float phi = cs*phibar;
      phibar = sn*phibar;
      float tau = sn*phi;

      //% Update x and w.

      //  t1 = phi / rho;
      //t2 = -theta / rho;
      //dk = (1 / rho)*w;
      float t1 = phi / rho;
      float t2 = -theta / rho;

      cusp::blas::copy(w, dk);
      cusp::blas::scal(dk, 1 / rho);

      //x = x + t1*w;
      //w = v + t2*w;
      //ddnorm = ddnorm + norm(dk) ^ 2;
      cusp::blas::axpy(w, x, t1);
      cusp::blas::axpby(v, w, w, 1, t2);
      ddnorm += cusp::blas::dot(dk, dk);
      //if wantvar, var = var + dk.*dk; end

      //  % Use a plane rotation on the right to eliminate the
      //  % super - diagonal element(theta) of the upper - bidiagonal matrix.
      //  % Then use the result to estimate  norm(x).

      //  delta = sn2*rho;
      //gambar = -cs2*rho;
      //rhs = phi - delta*z;
      //zbar = rhs / gambar;
      //xnorm = sqrt(xxnorm + zbar ^ 2);
      //gamma = norm([gambar theta]);
      float delta = sn2 * rho;
      float gambar = -cs2 * rho;
      float rhs = phi - delta * z;
      float zbar = rhs / gambar;
      xnorm = sqrt(xxnorm + zbar * zbar);
      float gamma = sqrt(gambar * gambar + theta * theta);
      //cs2 = gambar / gamma;
      //sn2 = theta / gamma;
      //z = rhs / gamma;
      //xxnorm = xxnorm + z ^ 2;
      cs2 = gambar / gamma;
      sn2 = theta / gamma;
      z = rhs / gamma;
      xxnorm = xxnorm + z * z;

      //% Test for convergence.
      //  % First, estimate the condition of the matrix  Abar,
      //  % and the norms of  rbar  and  Abar'rbar.

      //  Acond = Anorm*sqrt(ddnorm);
      //res1 = phibar ^ 2;
      //res2 = res2 + psi ^ 2;
      //rnorm = sqrt(res1 + res2);
      //Arnorm = alfa*abs(tau);
      Acond = Anorm * sqrt(ddnorm);
      float res1 = phibar * phibar;
      res2 = res2 + psi * psi;
      float rnorm = sqrt(res1 + res2);
      Arnorm = alfa * abs(tau);

      //% 07 Aug 2002:
      //% Distinguish between
      //  %    r1norm = || b - Ax || and
      //  %    r2norm = rnorm in current code
      //  % = sqrt(r1norm ^ 2 + damp ^ 2 * || x || ^ 2).
      //  %    Estimate r1norm from
      //  %    r1norm = sqrt(r2norm ^ 2 - damp ^ 2 * || x || ^ 2).
      //  % Although there is cancellation, it might be accurate enough.

      //  r1sq = rnorm ^ 2 - dampsq*xxnorm;
      //r1norm = sqrt(abs(r1sq));   if r1sq < 0, r1norm = -r1norm; end
      //  r2norm = rnorm;
      float r1sq = rnorm * rnorm - dampsq * xxnorm;
      r1norm = sqrt(abs(r1sq));
      if (r1sq < 0)
      {
        r1norm = -r1norm;
      }
      r2norm = rnorm;

      //% Now use these norms to estimate certain other quantities,
      //  % some of which will be small near a solution.

      //  test1 = rnorm / bnorm;
      //test2 = Arnorm / (Anorm*rnorm);
      //test3 = 1 / Acond;
      //t1 = test1 / (1 + Anorm*xnorm / bnorm);
      //rtol = btol + atol*Anorm*xnorm / bnorm;
      float test1 = rnorm / bnorm;
      float test2 = Arnorm / (Anorm * rnorm);
      float test3 = 1 / Acond;
      t1 = test1 / (1 + Anorm * xnorm / bnorm);
      float rtol = btol + atol * Anorm * xnorm / bnorm;

      //% The following tests guard against extremely small values of
      //  % atol, btol  or  ctol.  (The user may have set any or all of
      //  % the parameters  atol, btol, conlim  to 0.)
      //  % The effect is equivalent to the normal tests using
      //  % atol = eps, btol = eps, conlim = 1 / eps.

      //  if itn >= itnlim, istop = 7; end
      //    if 1 + test3 <= 1, istop = 6; end
      //      if 1 + test2 <= 1, istop = 5; end
      //        if 1 + t1 <= 1, istop = 4; end

      if (itn >= itnlim)
      {
        istop = 7;
      }
      if (1 + test3 <= 1)
      {
        istop = 6;
      }
      if (1 + test2 <= 1)
      {
        istop = 5;
      }
      if (1 + t1 <= 1)
      {
        istop = 4;
      }

      //          % Allow for tolerances set by the user.

      //          if  test3 <= ctol, istop = 3; end
      //            if  test2 <= atol, istop = 2; end
      //              if  test1 <= rtol, istop = 1; end

      //                % See if it is time to print something.
      if (test3 <= ctol)
      {
        istop = 3;
      }
      if (test2 <= atol)
      {
        istop = 2;
      }
      if (test1 <= rtol)
      {
        istop = 1;
      }

      //                prnt = 0;
      //if n <= 40, prnt = 1; end
      //  if itn <= 10, prnt = 1; end
      //    if itn >= itnlim - 10, prnt = 1; end
      //      if rem(itn, 10) == 0, prnt = 1; end
      //        if test3 <= 2 * ctol, prnt = 1; end
      //          if test2 <= 10 * atol, prnt = 1; end
      //            if test1 <= 10 * rtol, prnt = 1; end
      //              if istop ~= 0, prnt = 1; end
      /*bool prnt = false;
      if (n <= 40)
      {
      prnt = true;
      }
      if (itn <= 10)
      {
      prnt = true;
      }
      if ((itn % 10) == 0)
      {
      prnt = true;
      }
      if (test3 <= 2 * ctol)
      {
      prnt = true;
      }
      if (test2 <= 10 * atol)
      {
      prnt = true;
      }
      if (test1 <= 10 * rtol)
      {
      prnt = true;
      }
      if (istop != 0)
      {
      prnt = true;
      }*/

      //                if prnt
      //                  if show
      //                    str1 = sprintf('%6g %12.5e', itn, x(1));
      //str2 = sprintf(' %10.3e %10.3e', r1norm, r2norm);
      //str3 = sprintf('  %8.1e %8.1e', test1, test2);
      //str4 = sprintf(' %8.1e %8.1e', Anorm, Acond);
      //disp([str1 str2 str3 str4])
      //  end
      //  end
      //  if istop > 0, break, end
      //    end

      /*if (prnt)
      {

      }*/
      if (istop > 0)
      {
        break;
      }
      //    % End of iteration loop.
    }

    std::cout << "istop " << istop << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cusp::blas::axpby(sAndT.values, x, sAndTPlusX.values, 1, -1);

    for (int numPnt0 = 0; numPnt0 < numPoints; numPnt0 += maxNumPoints)
    {
      UnaryCostFunctionAndItsGradientWithRespectToParams3x6(
        thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&sPlusX(numPnt0, 0)),
        thrust::raw_pointer_cast(&tPlusX(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildeP(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacS(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacT(numPnt0, 0)),
        thrust::raw_pointer_cast(&sigma[numPnt0]),
        thrust::raw_pointer_cast(&e[numPnt0]),
        thrust::raw_pointer_cast(&jacE_.values[jacE_row_offsets[numPnt0]]),
        std::min(numPoints - numPnt0, maxNumPoints)
        );
    }

    for (int numPnt0 = 0; numPnt0 < numPairs; numPnt0 += maxNumPoints)
    {
      PairwiseCostFunctionAndItsGradientWithRespectToParamsWithPermutations3x12(
        thrust::raw_pointer_cast(&tildeP(0, 0)),
        thrust::raw_pointer_cast(&sPlusX(0, 0)),
        thrust::raw_pointer_cast(&tPlusX(0, 0)),
        thrust::raw_pointer_cast(&indPi[numPnt0]),
        thrust::raw_pointer_cast(&indPj[numPnt0]),
        thrust::raw_pointer_cast(&jacTildePi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTi(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTildePj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacSj(numPnt0, 0)),
        thrust::raw_pointer_cast(&jacTj(numPnt0, 0)),
        thrust::raw_pointer_cast(&ei[numPnt0]),
        thrust::raw_pointer_cast(&ej[numPnt0]),
        thrust::raw_pointer_cast(&jacE_.values[jacEi_row_offsets[numPnt0]]),
        thrust::raw_pointer_cast(&jacE_.values[jacEj_row_offsets[numPnt0]]),
        std::min(numPairs - numPnt0, maxNumPoints)
        );
    }

    std::cout << "Unary cost function " << (unaryCostFunction2 = cusp::blas::dot(e, e)) << std::endl;
    std::cout << "Pairwise cost function " << (pairwiseCostFunction2 = cusp::blas::dot(ei, ei) + cusp::blas::dot(ej, ej)) << std::endl;

    float numRho = (unaryCostFunction1 - unaryCostFunction2) + (pairwiseCostFunction1 - pairwiseCostFunction2);
    float denRho = (unaryCostFunction1 + pairwiseCostFunction1) - r1norm * r1norm;

    float rho = numRho / denRho;

    std::cout << "num(rho) " << numRho << std::endl;
    std::cout << "den(rho) " << denRho << std::endl;

    if ((rho < 0.01) || (numRho + 1 <= 1) || (denRho + 1 <= 1))
    {
      std::cout << "rho < pi1" << std::endl;
      lambda *= E;
      std::cout << "lambda = E lambda" << std::endl;

      if (lambda > 32000)
      {
        std::cout << "That is enough (" << iter << ")" << std::endl;
        break;
      }
    }
    else
    {
      cusp::copy(sAndTPlusX, sAndT);

      if (rho > 0.75)
      {
        std::cout << "rho > pi2" << std::endl;
        lambda *= D;
        std::cout << "lambda = D lambda" << std::endl;
      }
      else
      {
        std::cout << "pi1 < rho < pi2" << std::endl;
      }
    }

    std::cout << "rho " << rho << std::endl;
    std::cout << "lambda " << lambda << std::endl;
  }

  cusp::array2d<float, cusp::device_memory> p(numPoints, numDims);
  cusp::array2d<float, cusp::device_memory> jacP(numDims * numPoints, numParams);

  for (int numPnt0 = 0; numPnt0 < numPoints; numPnt0 += maxNumPoints)
  {
    AdjustLineEndpoints3(
      thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
      thrust::raw_pointer_cast(&s(numPnt0, 0)),
      thrust::raw_pointer_cast(&t(numPnt0, 0)),
      std::min(numPoints - numPnt0, maxNumPoints)
      );

    ProjectionOntoLineAndItsJacobian3x6(
      thrust::raw_pointer_cast(&tildeP(numPnt0, 0)),
      thrust::raw_pointer_cast(&s(numPnt0, 0)),
      thrust::raw_pointer_cast(&t(numPnt0, 0)),
      thrust::raw_pointer_cast(&jacTildeP(numPnt0, 0)),
      thrust::raw_pointer_cast(&jacS(numPnt0, 0)),
      thrust::raw_pointer_cast(&jacT(numPnt0, 0)),
      thrust::raw_pointer_cast(&p(numPnt0, 0)),
      thrust::raw_pointer_cast(&jacP(numPnt0, 0)),
      std::min(numPoints - numPnt0, maxNumPoints)
      );
  }

  cusp::copy(s, hS);
  cusp::copy(t, hT);

  array2d_view hP(numPoints, numDims, numDims, array1d_view(pP, pP + numDims * numPoints));
  cusp::copy(p, hP);
}
