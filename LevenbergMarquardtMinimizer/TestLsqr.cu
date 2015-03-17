#include "TestLsqr.h"
#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/blas.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/transpose.h>

void testLsqr()
{
  using std::abs;

  typedef cusp::array1d<float, cusp::host_memory> array1d;
  //typedef cusp::csr_matrix<int, float, cusp::host_memory> csrMatrix;
  typedef cusp::array2d<float, cusp::host_memory> csrMatrix;

  /*const size_t m = 5;
  const size_t n = 3;

  csrMatrix A(m, n, 0);
  array1d b(m);

  A(0, 0) = 2; A(0, 1) = 4; A(0, 2) =  5;
  A(1, 0) = 1; A(1, 1) = 3; A(1, 2) =  7;
  A(2, 0) = 3; A(2, 1) = 5; A(2, 2) = -1;
  A(3, 0) = 4; A(3, 1) = 7; A(3, 2) =  1;
  A(4, 0) = 3; A(4, 1) = 1; A(4, 2) =  2;

  b[0] = 12;
  b[1] = 17;
  b[2] = 13;
  b[3] = 21;
  b[4] =  7;
*/

  const size_t m = 4;
  const size_t n = 2;

  csrMatrix A(m, n, 0);
  array1d b(m);

  A(0, 0) = 2; A(0, 1) = -1;
  A(1, 0) = 5; A(1, 1) =  2; 
  A(2, 0) = 3; A(2, 1) = -1;
  A(3, 0) = 2; A(3, 1) = -3; 

  b[0] = 10;
  b[1] =  3;
  b[2] =  8;
  b[3] =  6;

  cusp::print(A);
  cusp::print(b);

  float atol = 0;
  float btol = 0;
  float damp = 0;
  int itnlim = 10;
  float conlim = 0;


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


  csrMatrix At;
  cusp::transpose(A, At);

  // Initialize.

  // Set up the first vectors u and v for the bidiagonalization.
  // These satisfy  beta*u = b, alfa*v = A'u.

  array1d u(A.num_rows);
  array1d v(A.num_cols);

  array1d Atu(A.num_cols);
  array1d Av(A.num_rows);

  array1d w(A.num_cols);
  array1d dk(A.num_cols);

  // u = b(1:m);        x = zeros(n, 1);
  cusp::copy(b, u);
  array1d x(A.num_cols, 0);

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
    Anorm = sqrt(Anorm * Anorm + alfa * alfa + beta * beta + damp * damp);
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
    bool prnt = false;
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
    }

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

    if (prnt)
    {

    }
    if (istop > 0)
    {
      break;
    }
    //    % End of iteration loop.

    cusp::print(x);
  }
  
  cusp::print(x);

  //    % Print the stopping condition.

  //    if show
  //      fprintf('\nlsqrSOL finished\n')
  //      disp(msg(istop + 1, :))
  //      disp(' ')
  //      str1 = sprintf('istop =%8g   r1norm =%8.1e', istop, r1norm);
  //str2 = sprintf('Anorm =%8.1e   Arnorm =%8.1e', Anorm, Arnorm);
  //str3 = sprintf('itn   =%8g   r2norm =%8.1e', itn, r2norm);
  //str4 = sprintf('Acond =%8.1e   xnorm  =%8.1e', Acond, xnorm);
  //disp([str1 '   ' str2])
  //  disp([str3 '   ' str4])
  //  disp(' ')
  //  end

  //  %---------------------------------------------------------------------- -
  //  % end function lsqrSOL
  //  %---------------------------------------------------------------------- -
}