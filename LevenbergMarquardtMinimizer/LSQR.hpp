#ifndef LSQR_hpp
#define LSQR_hpp

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LSQR   solves Ax = b or min ||Ax - b|| with or without damping,
// using the iterative algorithm of Paige and Saunders, ACM TOMS (1982).
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <algorithm>
#include <cmath>
#include <cusp/blas/blas.h>
#include <cusp/copy.h>
#include <cusp/multiply.h>
#include <iomanip>
#include <glog/logging.h>

template<typename ValueType>
static ValueType d2norm(ValueType a, ValueType b)
{
  //-------------------------------------------------------------------
  // d2norm returns sqrt( a**2 + b**2 )
  // with precautions to avoid overflow.
  //-------------------------------------------------------------------
  ValueType scale = std::abs(a) + std::abs(b);

  if (scale == 0)
  {
    return 0;
  }
  else
  {
    ValueType ascale = a / scale;
    ValueType bscale = b / scale;
    return scale * std::sqrt(ascale * ascale + bscale * bscale);
  }
}

template<typename LinearOperator1, typename LinearOperator2, typename Vector1, typename Vector2, typename ValueType = typename LinearOperator1::value_type>
void LSQR(const LinearOperator1& A, const LinearOperator2& At, const Vector1& b, ValueType damp, Vector2& x,
  ValueType atol, ValueType btol, ValueType conlim, int itnlim,
  int& istop, int& itn, ValueType& Anorm, ValueType& Acond, ValueType& rnorm, ValueType& Arnorm, ValueType& xnorm)
{
  //-------------------------------------------------------------------
  // LSQR  finds a solution x to the following problems:
  //
  // 1. Unsymmetric equations:    Solve  A*x = b
  //
  // 2. Linear least squares:     Solve  A*x = b
  //                              in the least-squares sense
  //
  // 3. Damped least squares:     Solve  (   A    )*x = ( b )
  //                                     ( damp*I )     ( 0 )
  //                              in the least-squares sense
  //
  // where A is a matrix with m rows and n columns, b is an m-vector,
  // and damp is a scalar.  (All quantities are real.)
  // The matrix A is treated as a linear operator.  It is accessed
  // by means of subroutine calls with the following purpose:
  //
  // call Aprod1(m,n,x,y)  must compute y = y + A*x  without altering x.
  // call Aprod2(m,n,x,y)  must compute x = x + A'*y without altering y.
  //
  // LSQR uses an iterative method to approximate the solution.
  // The number of iterations required to reach a certain accuracy
  // depends strongly on the scaling of the problem.  Poor scaling of
  // the rows or columns of A should therefore be avoided where
  // possible.
  //
  // For example, in problem 1 the solution is unaltered by
  // row-scaling.  If a row of A is very small or large compared to
  // the other rows of A, the corresponding row of ( A  b ) should be
  // scaled up or down.
  //
  // In problems 1 and 2, the solution x is easily recovered
  // following column-scaling.  Unless better information is known,
  // the nonzero columns of A should be scaled so that they all have
  // the same Euclidean norm (e.g., 1.0).
  //
  // In problem 3, there is no freedom to re-scale if damp is
  // nonzero.  However, the value of damp should be assigned only
  // after attention has been paid to the scaling of A.
  //
  // The parameter damp is intended to help regularize
  // ill-conditioned systems, by preventing the true solution from
  // being very large.  Another aid to regularization is provided by
  // the parameter Acond, which may be used to terminate iterations
  // before the computed solution becomes very large.
  //
  // Note that x is not an input parameter.
  // If some initial estimate x0 is known and if damp = 0,
  // one could proceed as follows:
  //
  // 1. Compute a residual vector     r0 = b - A*x0.
  // 2. Use LSQR to solve the system  A*dx = r0.
  // 3. Add the correction dx to obtain a final solution x = x0 + dx.
  //
  // This requires that x0 be available before and after the call
  // to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
  // to solve A*x = b and k2 iterations to solve A*dx = r0.
  // If x0 is "good", norm(r0) will be smaller than norm(b).
  // If the same stopping tolerances atol and btol are used for each
  // system, k1 and k2 will be similar, but the final solution x0 + dx
  // should be more accurate.  The only way to reduce the total work
  // is to use a larger stopping tolerance for the second system.
  // If some value btol is suitable for A*x = b, the larger value
  // btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
  //
  // Preconditioning is another way to reduce the number of iterations.
  // If it is possible to solve a related system M*x = b efficiently,
  // where M approximates A in some helpful way
  // (e.g. M - A has low rank or its elements are small relative to
  // those of A), LSQR may converge more rapidly on the system
  //       A*M(inverse)*z = b,
  // after which x can be recovered by solving M*x = z.
  //
  // NOTE: If A is symmetric, LSQR should not be used!
  // Alternatives are the symmetric conjugate-gradient method (CG)
  // and/or SYMMLQ.
  // SYMMLQ is an implementation of symmetric CG that applies to
  // any symmetric A and will converge more rapidly than LSQR.
  // If A is positive definite, there are other implementations of
  // symmetric CG that require slightly less work per iteration
  // than SYMMLQ (but will take the same number of iterations).
  //
  //
  // Notation
  // --------
  // The following quantities are used in discussing the subroutine
  // parameters:
  //
  // Abar   =  (  A   ),        bbar  =  (b)
  //           (damp*I)                  (0)
  //
  // r      =  b - A*x,         rbar  =  bbar - Abar*x
  //
  // rnorm  =  sqrt( norm(r)**2  +  damp**2 * norm(x)**2 )
  //        =  norm( rbar )
  //
  // eps    =  the relative precision of floating-point arithmetic.
  //           On most machines, eps is about 1.0e-7 and 1.0e-16
  //           in single and double precision respectively.
  //           We expect eps to be about 1e-16 always.
  //
  // LSQR  minimizes the function rnorm with respect to x.
  //
  //
  // Parameters
  // ----------
  // m       input      m, the number of rows in A.
  //
  // n       input      n, the number of columns in A.
  //
  // Aprod1, Aprod2     See above.
  //
  // damp    input      The damping parameter for problem 3 above.
  //                    (damp should be 0.0 for problems 1 and 2.)
  //                    If the system A*x = b is incompatible, values
  //                    of damp in the range 0 to sqrt(eps)*norm(A)
  //                    will probably have a negligible effect.
  //                    Larger values of damp will tend to decrease
  //                    the norm of x and reduce the number of 
  //                    iterations required by LSQR.
  //
  //                    The work per iteration and the storage needed
  //                    by LSQR are the same for all values of damp.
  //
  // b(m)    input      The rhs vector b.
  //
  // x(n)    output     Returns the computed solution x.
  //
  // atol    input      An estimate of the relative error in the data
  //                    defining the matrix A.  For example, if A is
  //                    accurate to about 6 digits, set atol = 1.0e-6.
  //
  // btol    input      An estimate of the relative error in the data
  //                    defining the rhs b.  For example, if b is
  //                    accurate to about 6 digits, set btol = 1.0e-6.
  //
  // conlim  input      An upper limit on cond(Abar), the apparent
  //                    condition number of the matrix Abar.
  //                    Iterations will be terminated if a computed
  //                    estimate of cond(Abar) exceeds conlim.
  //                    This is intended to prevent certain small or
  //                    zero singular values of A or Abar from
  //                    coming into effect and causing unwanted growth
  //                    in the computed solution.
  //
  //                    conlim and damp may be used separately or
  //                    together to regularize ill-conditioned systems.
  //
  //                    Normally, conlim should be in the range
  //                    1000 to 1/eps.
  //                    Suggested value:
  //                    conlim = 1/(100*eps)  for compatible systems,
  //                    conlim = 1/(10*sqrt(eps)) for least squares.
  //
  //         Note: Any or all of atol, btol, conlim may be set to zero.
  //         The effect will be the same as the values eps, eps, 1/eps.
  //
  // itnlim  input      An upper limit on the number of iterations.
  //                    Suggested value:
  //                    itnlim = n/2   for well-conditioned systems
  //                                   with clustered singular values,
  //                    itnlim = 4*n   otherwise.
  //
  // nout    input      File number for printed output.  If positive,
  //                    a summary will be printed on file nout.
  //
  // istop   output     An integer giving the reason for termination:
  //
  //            0       x = 0  is the exact solution.
  //                    No iterations were performed.
  //
  //            1       The equations A*x = b are probably compatible.
  //                    Norm(A*x - b) is sufficiently small, given the
  //                    values of atol and btol.
  //
  //            2       damp is zero.  The system A*x = b is probably
  //                    not compatible.  A least-squares solution has
  //                    been obtained that is sufficiently accurate,
  //                    given the value of atol.
  //
  //            3       damp is nonzero.  A damped least-squares
  //                    solution has been obtained that is sufficiently
  //                    accurate, given the value of atol.
  //
  //            4       An estimate of cond(Abar) has exceeded conlim.
  //                    The system A*x = b appears to be ill-conditioned,
  //                    or there could be an error in Aprod1 or Aprod2.
  //
  //            5       The iteration limit itnlim was reached.
  //
  // itn     output     The number of iterations performed.
  //
  // Anorm   output     An estimate of the Frobenius norm of Abar.
  //                    This is the square-root of the sum of squares
  //                    of the elements of Abar.
  //                    If damp is small and the columns of A
  //                    have all been scaled to have length 1.0,
  //                    Anorm should increase to roughly sqrt(n).
  //                    A radically different value for Anorm may
  //                    indicate an error in Aprod1 or Aprod2.
  //
  // Acond   output     An estimate of cond(Abar), the condition
  //                    number of Abar.  A very high value of Acond
  //                    may again indicate an error in Aprod1 or Aprod2.
  //
  // rnorm   output     An estimate of the final value of norm(rbar),
  //                    the function being minimized (see notation
  //                    above).  This will be small if A*x = b has
  //                    a solution.
  //
  // Arnorm  output     An estimate of the final value of
  //                    norm( Abar(transpose)*rbar ), the norm of
  //                    the residual for the normal equations.
  //                    This should be small in all cases.  (Arnorm
  //                    will often be smaller than the true value
  //                    computed from the output vector x.)
  //
  // xnorm   output     An estimate of norm(x) for the final solution x.
  //
  // Precision
  // ---------
  // The number of iterations required by LSQR will decrease
  // if the computation is performed in higher precision.
  // At least 15-digit arithmetic should normally be used.
  // "real(dp)" declarations should normally be 8-byte words.
  // If this ever changes, the BLAS routines  dnrm2, dscal
  // (Lawson, et al., 1979) will also need to be changed.
  //
  //
  // References
  // ----------
  // C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
  //      linear equations and sparse least squares,
  //      ACM Transactions on Mathematical Software 8, 1 (March 1982),
  //      pp. 43-71.
  //
  // C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
  //      linear equations and least-squares problems,
  //      ACM Transactions on Mathematical Software 8, 2 (June 1982),
  //      pp. 195-209.
  //
  // C.L. Lawson, R.J. Hanson, D.R. Kincaid and F.T. Krogh,
  //      Basic linear algebra subprograms for Fortran usage,
  //      ACM Transactions on Mathematical Software 5, 3 (Sept 1979),
  //      pp. 308-323 and 324-325.
  //-------------------------------------------------------------------

  // Local arrays and variables
  typedef typename LinearOperator1::memory_space MemorySpace;
  typedef cusp::array1d<ValueType, MemorySpace> ArrayType;
  
  ArrayType u(A.num_rows), v(A.num_cols), w(A.num_cols);
  ArrayType tmpU(u.size()), tmpV(v.size());
  
  bool damped, prnt;
  int maxdx, nconv, nstop;
  ValueType alfopt,
    alpha, beta, bnorm, cs, cs1, cs2, ctol,
    delta, dknorm, dnorm, dxk, dxmax, gamma,
    gambar, phi, phibar, psi, res2, rho,
    rhobar, rhbar1, rhs, rtol, sn, sn1, sn2,
    tau, temp, test1, test2, test3, theta,
    t1, t2, t3, xnorm1, z, zbar;

  // Local constants
  const char* msg[] = {
    "The exact solution is  x = 0                         ",
    "A solution to Ax = b was found, given atol, btol     ",
    "A least-squares solution was found, given atol       ",
    "A damped least-squares solution was found, given atol",
    "Cond(Abar) seems to be too large, given conlim       ",
    "The iteration limit was reached                      "
  };
  //-------------------------------------------------------------------

  // Initialize.

  damped = damp > 0;
  itn = 0;
  istop = 0;
  nstop = 0;
  maxdx = 0;
  ctol = 0;
  if (conlim > 0) ctol = 1 / conlim;
  Anorm = 0;
  Acond = 0;
  dnorm = 0;
  dxmax = 0;
  res2 = 0;
  psi = 0;
  xnorm = 0;
  xnorm1 = 0;
  cs2 = -1;
  sn2 = 0;
  z = 0;

  LOG(INFO) << "";
  LOG(INFO) << "";
  LOG(INFO) << " LSQR      --      Least-squares solution of  Ax = b";

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << " The matrix A has" << std::setw(9) << A.num_rows << " rows and" << std::setw(9) << A.num_cols << " columns";
  LOG_IF(INFO, damped) << std::setfill(' ') << std::scientific << " The damping parameter is         damp   =" << std::setw(10) << std::setprecision(2) << damp;

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << std::scientific << " atol   =" << std::setw(10) << std::setprecision(2) << atol << std::setw(15) << "" << "conlim =" << std::setw(10) << std::setprecision(2) << conlim;
  LOG(INFO) << std::setfill(' ') << std::scientific << " btol   =" << std::setw(10) << std::setprecision(2) << btol << std::setw(15) << "" << "itnlim =" << std::setw(10) << itnlim;

  //-------------------------------------------------------------------
  // Set up the first vectors u and v for the bidiagonalization.
  // These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
  //-------------------------------------------------------------------
  cusp::copy(b, u);
  cusp::blas::fill(v, 0);
  cusp::blas::fill(x, 0);

  alpha = 0;
  beta = cusp::blas::nrm2(u);

  if (beta > 0)
  {
    cusp::blas::scal(u, 1 / beta);
    cusp::multiply(At, u, v); // v = A'*u
    alpha = cusp::blas::nrm2(v);
  }

  if (alpha > 0)
  {
    cusp::blas::scal(v, 1 / alpha);
    cusp::copy(v, w);
  }

  Arnorm = alpha * beta;
  if (Arnorm == 0) goto label800;

  rhobar = alpha;
  phibar = beta;
  bnorm = beta;
  rnorm = beta;

  if (damped)
  {
    LOG(INFO) << "   Itn       x(1)           Function" << "     Compatible   LS     Norm Abar Cond Abar alfa_opt";
  }
  else
  {
    LOG(INFO) << "   Itn       x(1)           Function" << "     Compatible   LS        Norm A    Cond A";
  }
  test1 = 1;
  test2 = alpha / beta;
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(6) << itn << std::setw(17) << std::setprecision(9) << x[0] << std::setw(17) << std::setprecision(9) << rnorm << std::setw(10) << std::setprecision(2) << test1 << std::setw(10) << std::setprecision(2) << test2;

  //===================================================================
  // Main iteration loop.
  //===================================================================
  while (true)
  {
    itn = itn + 1;

    //----------------------------------------------------------------
    //Perform the next step of the bidiagonalization to obtain the
    //next beta, u, alpha, v.These satisfy
    //beta*u = A*v - alpha*u,
    //alpha*v = A'*u -  beta*v.
    //----------------------------------------------------------------
    cusp::multiply(A, v, tmpU);
    cusp::blas::axpy(u, tmpU, -alpha);
    u.swap(tmpU);
    beta = cusp::blas::nrm2(u);

    // Accumulate Anorm = || Bk || = norm([alpha beta damp]).

    temp = d2norm(alpha, beta);
    temp = d2norm(temp, damp);
    Anorm = d2norm(Anorm, temp);

    if (beta > 0)
    {
      cusp::blas::scal(u, 1 / beta);
      cusp::multiply(At, u, tmpV);
      cusp::blas::axpy(v, tmpV, -beta);
      v.swap(tmpV);
      alpha = cusp::blas::nrm2(v);
      if (alpha > 0)
      {
        cusp::blas::scal(v, 1 / alpha);
      }
    }

    //----------------------------------------------------------------
    // Use a plane rotation to eliminate the damping parameter.
    // This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
    //----------------------------------------------------------------
    rhbar1 = rhobar;
    if (damped)
    {
      rhbar1 = d2norm(rhobar, damp);
      cs1 = rhobar / rhbar1;
      sn1 = damp / rhbar1;
      psi = sn1 * phibar;
      phibar = cs1 * phibar;
    }

    //----------------------------------------------------------------
    // Use a plane rotation to eliminate the subdiagonal element (beta)
    // of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
    //----------------------------------------------------------------
    rho = d2norm(rhbar1, beta);
    cs = rhbar1 / rho;
    sn = beta / rho;
    theta = sn * alpha;
    rhobar = -cs * alpha;
    phi = cs * phibar;
    phibar = sn * phibar;
    tau = sn * phi;

    //----------------------------------------------------------------
    // Update  x, w  and (perhaps) the standard error estimates.
    // ---------------------------------------------------------------
    t1 = phi / rho;
    t2 = -theta / rho;
    t3 = 1 / rho;

    dknorm = (t3 * t3) * cusp::blas::dot(w, w);
    cusp::blas::axpy(w, x, t1);
    cusp::blas::axpby(v, w, w, 1, t2);

    //----------------------------------------------------------------
    // Monitor the norm of d_k, the update to x.
    // dknorm = norm( d_k )
    // dnorm  = norm( D_k ),       where   D_k = (d_1, d_2, ..., d_k )
    // dxk    = norm( phi_k d_k ), where new x = x_k + phi_k d_k.
    //----------------------------------------------------------------
    dknorm = std::sqrt(dknorm);
    dnorm = d2norm(dnorm, dknorm);
    dxk = std::abs(phi * dknorm);
    if (dxmax < dxk)
    {
      dxmax = dxk;
      maxdx = itn;
    }

    //----------------------------------------------------------------
    // Use a plane rotation on the right to eliminate the
    // super-diagonal element (theta) of the upper-bidiagonal matrix.
    // Then use the result to estimate  norm(x).
    //----------------------------------------------------------------
    delta = sn2 * rho;
    gambar = -cs2 * rho;
    rhs = phi - delta * z;
    zbar = rhs / gambar;
    xnorm = d2norm(xnorm1, zbar);
    gamma = d2norm(gambar, theta);
    cs2 = gambar / gamma;
    sn2 = theta / gamma;
    z = rhs / gamma;
    xnorm1 = d2norm(xnorm1, z);

    //----------------------------------------------------------------
    // Test for convergence.
    // First, estimate the norm and condition of the matrix  Abar,
    // and the norms of  rbar  and  Abar(transpose)*rbar.
    //----------------------------------------------------------------
    Acond = Anorm * dnorm;
    res2 = d2norm(res2, psi);
    rnorm = d2norm(res2, phibar);
    rnorm = rnorm + 1e-30; // Prevent rnorm == 0.0
    Arnorm = alpha * std::abs(tau);

    // Now use these norms to estimate certain other quantities,
    // some of which will be small near a solution.

    alfopt = sqrt(rnorm / (dnorm * xnorm));
    test1 = rnorm / bnorm;
    test2 = 0;
    test2 = Arnorm / (Anorm * rnorm);
    test3 = 1 / Acond;
    t1 = test1 / (1 + Anorm * xnorm / bnorm);
    rtol = btol + atol * Anorm * xnorm / bnorm;

    // The following tests guard against extremely small values of
    // atol, btol  or  ctol.  (The user may have set any or all of
    // the parameters  atol, btol, conlim  to zero.)
    // The effect is equivalent to the normal tests using
    // atol = eps,  btol = eps,  conlim = 1/eps.

    t3 = 1 + test3;
    t2 = 1 + test2;
    t1 = 1 + t1;
    if (itn >= itnlim) istop = 5;
    if (t3 <= 1) istop = 4;
    if (t2 <= 1) istop = 2;
    if (t1 <= 1) istop = 1;

    // Allow for tolerances set by the user.

    if (test3 <= ctol) istop = 4;
    if (test2 <= atol) istop = 2;
    if (test1 <= rtol) istop = 1;

    //----------------------------------------------------------------
    // See if it is time to print something.
    //----------------------------------------------------------------
    prnt = false;
    if (A.num_cols <= 40) prnt = true;
    if (itn <= 10) prnt = true;
    if (itn >= itnlim - 10) prnt = true;
    if (itn % 10 == 0) prnt = true;
    if (test3 <= 2.0 * ctol) prnt = true;
    if (test2 <= 10.0 * atol) prnt = true;
    if (test1 <= 10.0 * rtol) prnt = true;
    if (istop != 0) prnt = true;

    if (prnt) // Print a line for this iteration.
    {
      LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(6) << itn << std::setw(17) << std::setprecision(9) << x[0] << std::setw(17) << std::setprecision(9) << rnorm << std::setw(10) << std::setprecision(2) << std::setw(10) << test1 << std::setw(10) << test2 << std::setw(10) << Anorm << std::setw(10) << Acond << std::setw(9) << std::setprecision(1) << alfopt;
    }

    //----------------------------------------------------------------
    // Stop if appropriate.
    // The convergence criteria are required to be met on  nconv
    // consecutive iterations, where  nconv  is set below.
    // Suggested value:  nconv = 1, 2  or  3.
    //----------------------------------------------------------------
    if (istop == 0)
    {
      nstop = 0;
    }
    else
    {
      nconv = 1;
      nstop = nstop + 1;
      if (nstop < nconv && itn < itnlim) istop = 0;
    }
    if (istop != 0) break;

  }
  //===================================================================
  // End of iteration loop.
  //===================================================================

  // Come here if Arnorm = 0, or if normal exit.
label800:
  if (damped && istop == 2) istop = 3;

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "istop  =" << std::setw(2) << istop << std::setw(15) << "" << "itn    =" << std::setw(8) << itn;
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "Anorm  =" << std::setw(12) << std::setprecision(5) << Anorm << std::setw(5) << "" << "Acond  =" << std::setw(12) << std::setprecision(5) << Acond;
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "bnorm  =" << std::setw(12) << std::setprecision(5) << bnorm << std::setw(5) << "" << "xnorm  =" << std::setw(12) << std::setprecision(5) << xnorm;
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "rnorm  =" << std::setw(12) << std::setprecision(5) << rnorm << std::setw(5) << "" << "Arnorm =" << std::setw(12) << std::setprecision(5) << Arnorm;

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "max dx =" << std::setw(9) << std::setprecision(1) << dxmax << " occurred at itn" << std::setw(8) << maxdx;
  LOG(INFO) << std::setfill(' ') << std::scientific << std::setw(5) << "" << "       =" << std::setw(9) << std::setprecision(1) << dxmax / (xnorm + 1.0e-30) << " * xnorm";

  LOG(INFO) << "";
  LOG(INFO) << std::setfill(' ') << std::setw(5) << "" << msg[istop - 1];
  LOG(INFO) << "";
}

#endif//LSQR_hpp