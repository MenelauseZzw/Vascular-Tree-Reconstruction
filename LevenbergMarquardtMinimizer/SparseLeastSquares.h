#ifndef SparseLeastSquares_h
#define SparseLeastSquares_h

#include <algorithm>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/blas/blas.h>
#include <cusp/multiply.h>
#include <iomanip>
#include <glog/logging.h>
#include <sstream>
#include <string>

template<typename LinearOperator>
class SparseLeastSquares
{
public:
  typedef typename LinearOperator::value_type ValueType;
  typedef typename LinearOperator::memory_space MemorySpace;

  typedef cusp::array1d<ValueType, MemorySpace> Array1d;
  typedef cusp::array2d<ValueType, MemorySpace> Array2d;

  SparseLeastSquares(const LinearOperator& A, const LinearOperator& At, const Array1d& b)
    : A(A), At(At), b(b)
  {
  }

  ValueType r1norm;

  void Solve(Array1d& x, ValueType atol, ValueType btol, ValueType conlim, ValueType damp, int itnlim, bool show)
  {
    // Initialize.

    // msg = ['The exact solution is  x = 0                              '
    //   'Ax - b is small enough, given atol, btol                  '
    //   'The least-squares solution is good enough, given atol     '
    //   'The estimate of cond(Abar) has exceeded conlim            '
    //   'Ax - b is small enough for this machine                   '
    //   'The least-squares solution is good enough for this machine'
    //   'Cond(Abar) seems to be too large for this machine         '
    //   'The iteration limit has been reached                      '];
    const char* msg[] = {
      "The exact solution is  x = 0                              ",
      "Ax - b is small enough, given atol, btol                  ",
      "The least-squares solution is good enough, given atol     ",
      "The estimate of cond(Abar) has exceeded conlim            ",
      "Ax - b is small enough for this machine                   ",
      "The least-squares solution is good enough for this machine",
      "Cond(Abar) seems to be too large for this machine         ",
      "The iteration limit has been reached                      "
    };

    // itn = 0;             istop = 0;
    int itn = 0;
    int istop = 0;

    // ctol   = 0;             if conlim > 0, ctol = 1/conlim; end;
    ValueType ctol = 0;
    if (conlim > 0) ctol = 1 / conlim;

    // Anorm  = 0;             Acond  = 0;
    ValueType Anorm = 0;
    ValueType Acond = 0;

    // dampsq = damp^2;        ddnorm = 0;             res2   = 0;
    ValueType dampsq = damp * damp;
    ValueType ddnorm = 0;
    ValueType res2 = 0;

    // xnorm  = 0;             xxnorm = 0;             z      = 0;
    ValueType xnorm = 0;
    ValueType xxnorm = 0;
    ValueType z = 0;

    // cs2    = -1;            sn2    = 0;
    ValueType cs2 = -1;
    ValueType sn2 = 0;

    // Set up the first vectors u and v for the bidiagonalization.
    // These satisfy  beta*u = b,  alfa*v = A'u.

    // u      = b(1:m);        x    = zeros(n,1);
    Array1d u(b);
    cusp::blas::fill(x, 0);

    // alfa   = 0;             beta = norm(u);
    ValueType alfa = 0;
    ValueType beta = cusp::blas::nrm2(u);

    Array1d v(A.num_cols);
    // if beta > 0
    //   u = (1 / beta)*u;
    //   v = A'*u;
    //   alfa = norm(v);
    // end
    if (beta > 0)
    {
      cusp::blas::scal(u, 1 / beta);
      cusp::multiply(At, u, v);
      alfa = cusp::blas::nrm2(v);
    }

    Array1d w(A.num_cols);
    // if alfa > 0
    //   v = (1 / alfa)*v;      w = v;
    // end
    if (alfa > 0)
    {
      cusp::blas::scal(v, 1 / alfa);
      cusp::copy(v, w);
    }

    // Arnorm = alfa*beta;     if Arnorm == 0, disp(msg(1, :)); return, end
    ValueType Arnorm = alfa * beta;
    if (Arnorm == 0)
    {
      LOG(INFO) << msg[1 - 1];
      return;
    }

    // rhobar = alfa;          phibar = beta;          bnorm = beta;
    ValueType rhobar = alfa;
    ValueType phibar = beta;
    ValueType bnorm = beta;

    // rnorm = beta;
    ValueType rnorm = beta;

    // r1norm = rnorm;
    r1norm = rnorm;

    // r2norm = rnorm;
    ValueType r2norm = rnorm;

    // head1 = '   Itn      x(1)       r1norm     r2norm ';
    // head2 = ' Compatible   LS      Norm A   Cond A';
    const char* head1 = "   Itn      x(1)       r1norm     r2norm ";
    const char* head2 = " Compatible   LS      Norm A   Cond A";

    // if show
    if (show)
    {
      // disp(' ')
      // disp([head1 head2])
      LOG(INFO) << " ";
      LOG(INFO) << head1 << head2;

      // test1 = 1;          test2 = alfa / beta;
      ValueType test1 = 1;
      ValueType test2 = alfa / beta;

      // str1 = sprintf('%6g %12.5e', itn, x(1));
      std::string str1;
      {
        std::ostringstream sstream;
        sstream << std::scientific << std::setw(6) << itn << " " << std::setw(12) << std::setprecision(5) << x[1 - 1];
        str1 = sstream.str();
      }

      // str2 = sprintf(' %10.3e %10.3e', r1norm, r2norm);
      std::string str2;
      {
        std::ostringstream sstream;
        sstream << std::scientific << " " << std::setw(10) << std::setprecision(3) << r1norm << " " << std::setw(10) << std::setprecision(3) << r2norm;
        str2 = sstream.str();
      }

      // str3 = sprintf('  %8.1e %8.1e', test1, test2);
      std::string str3;
      {
        std::ostringstream sstream;
        sstream << std::scientific << "  " << std::setw(8) << std::setprecision(1) << test1 << " " << std::setw(8) << std::setprecision(1) << test2;
        str3 = sstream.str();
      }

      // disp([str1 str2 str3])
      LOG(INFO) << str1 << str2 << str3;
    }
    // end

    //------------------------------------------------------------------
    //     Main iteration loop.
    //------------------------------------------------------------------
    Array1d tmpU(u.size());
    Array1d tmpV(v.size());
    Array1d dk(w.size());

    // while itn < itnlim
    while (itn < itnlim)
    {
      // itn = itn + 1;
      itn = itn + 1;

      // Perform the next step of the bidiagonalization to obtain the
      // next beta, u, alfa, v.These satisfy the relations
      //      beta*u = A*v - alfa*u,
      //      alfa*v = A'*u - beta*v.

      // u = A*v    - alfa*u;
      cusp::multiply(A, v, tmpU);
      cusp::blas::axpy(u, tmpU, -alfa);
      u.swap(tmpU);

      // beta = norm(u);
      beta = cusp::blas::nrm2(u);

      // if beta > 0
      if (beta > 0)
      {
        // u = (1 / beta)*u;
        cusp::blas::scal(u, 1 / beta);

        // Anorm = norm([Anorm alfa beta damp]);
        Anorm = sqrt(Anorm * Anorm + alfa * alfa + beta * beta + damp * damp);

        // v = A'*u   - beta*v;
        cusp::multiply(At, u, tmpV);
        cusp::blas::axpy(v, tmpV, -beta);
        v.swap(tmpV);

        // alfa = norm(v);
        // if alfa > 0, v = (1 / alfa)*v; end
        alfa = cusp::blas::nrm2(v);
        if (alfa > 0) cusp::blas::scal(v, 1 / alfa);
      }
      // end

      // Use a plane rotation to eliminate the damping parameter.
      // This alters the diagonal(rhobar) of the lower - bidiagonal matrix.

      // rhobar1 = norm([rhobar damp]);
      // cs1 = rhobar / rhobar1;
      // sn1 = damp / rhobar1;
      // psi = sn1*phibar;
      // phibar = cs1*phibar;
      ValueType rhobar1 = sqrt(rhobar * rhobar + damp * damp);
      ValueType cs1 = rhobar / rhobar1;
      ValueType sn1 = damp / rhobar1;
      ValueType psi = sn1 * phibar;
      phibar = cs1 * phibar;

      // Use a plane rotation to eliminate the subdiagonal element(beta)
      // of the lower - bidiagonal matrix, giving an upper - bidiagonal matrix.

      // rho = norm([rhobar1 beta]);
      // cs = rhobar1 / rho;
      // sn = beta / rho;
      // theta = sn*alfa;
      // rhobar = -cs*alfa;
      // phi = cs*phibar;
      // phibar = sn*phibar;
      // tau = sn*phi;
      ValueType rho = sqrt(rhobar1 * rhobar1 + beta * beta);
      ValueType cs = rhobar1 / rho;
      ValueType sn = beta / rho;
      ValueType theta = sn * alfa;
      rhobar = -cs * alfa;
      ValueType phi = cs * phibar;
      phibar = sn * phibar;
      ValueType tau = sn * phi;

      // Update x and w.

      // t1 = phi / rho;
      // t2 = -theta / rho;
      ValueType t1 = phi / rho;
      ValueType t2 = -theta / rho;

      // dk = (1 / rho)*w;
      cusp::blas::copy(w, dk);
      cusp::blas::scal(dk, 1 / rho);

      // x = x + t1*w;
      // w = v + t2*w;
      cusp::blas::axpy(w, x, t1);
      cusp::blas::axpby(v, w, w, 1, t2);

      // ddnorm = ddnorm + norm(dk) ^ 2;
      ddnorm = ddnorm + cusp::blas::dot(dk, dk);

      // Use a plane rotation on the right to eliminate the
      // super - diagonal element(theta) of the upper - bidiagonal matrix.
      // Then use the result to estimate  norm(x).

      // delta = sn2*rho;
      // gambar = -cs2*rho;
      // rhs = phi - delta*z;
      // zbar = rhs / gambar;
      // xnorm = sqrt(xxnorm + zbar ^ 2);
      // gamma = norm([gambar theta]);
      // cs2 = gambar / gamma;
      // sn2 = theta / gamma;
      // z = rhs / gamma;
      // xxnorm = xxnorm + z ^ 2;
      ValueType delta = sn2 * rho;
      ValueType gambar = -cs2 * rho;
      ValueType rhs = phi - delta * z;
      ValueType zbar = rhs / gambar;
      xnorm = sqrt(xxnorm + zbar * zbar);
      ValueType gamma = sqrt(gambar * gambar + theta * theta);
      cs2 = gambar / gamma;
      sn2 = theta / gamma;
      z = rhs / gamma;
      xxnorm = xxnorm + z * z;

      // Test for convergence.
      // First, estimate the condition of the matrix  Abar,
      // and the norms of  rbar  and  Abar'rbar.

      // Acond = Anorm*sqrt(ddnorm);
      // res1 = phibar ^ 2;
      // res2 = res2 + psi ^ 2;
      // rnorm = sqrt(res1 + res2);
      // Arnorm = alfa*abs(tau);
      Acond = Anorm * sqrt(ddnorm);
      ValueType res1 = phibar * phibar;
      res2 = res2 + psi * psi;
      ValueType rnorm = sqrt(res1 + res2);
      Arnorm = alfa * abs(tau);

      // 07 Aug 2002:
      // Distinguish between
      //    r1norm = || b - Ax || and
      //    r2norm = rnorm in current code
      //           = sqrt(r1norm ^ 2 + damp ^ 2 * || x || ^ 2).
      //    Estimate r1norm from
      //    r1norm = sqrt(r2norm ^ 2 - damp ^ 2 * || x || ^ 2).
      // Although there is cancellation, it might be accurate enough.

      // r1sq = rnorm ^ 2 - dampsq*xxnorm;
      // r1norm = sqrt(abs(r1sq));   if r1sq < 0, r1norm = -r1norm; end
      // r2norm = rnorm;
      ValueType r1sq = rnorm * rnorm - dampsq * xxnorm;
      r1norm = sqrt(abs(r1sq)); if (r1sq < 0) r1norm = -r1norm;
      r2norm = rnorm;

      // Now use these norms to estimate certain other quantities,
      // some of which will be small near a solution.

      // test1 = rnorm / bnorm;
      // test2 = Arnorm / (Anorm*rnorm);
      // test3 = 1 / Acond;
      // t1 = test1 / (1 + Anorm*xnorm / bnorm);
      // rtol = btol + atol*Anorm*xnorm / bnorm;
      ValueType test1 = rnorm / bnorm;
      ValueType test2 = Arnorm / (Anorm * rnorm);
      ValueType test3 = 1 / Acond;
      t1 = test1 / (1 + Anorm * xnorm / bnorm);
      ValueType rtol = btol + atol * Anorm * xnorm / bnorm;

      // The following tests guard against extremely small values of
      // atol, btol  or  ctol.  (The user may have set any or all of
      // the parameters  atol, btol, conlim  to 0.)
      // The effect is equivalent to the normal tests using
      // atol = eps, btol = eps, conlim = 1 / eps.

      // if itn >= itnlim, istop = 7; end
      // if 1 + test3 <= 1, istop = 6; end
      // if 1 + test2 <= 1, istop = 5; end
      // if 1 + t1 <= 1, istop = 4; end
      if (itn >= itnlim) istop = 7;
      if (1 + test3 <= 1) istop = 6;
      if (1 + test2 <= 1) istop = 5;
      if (1 + t1 <= 1) istop = 4;

      // Allow for tolerances set by the user.

      // if  test3 <= ctol, istop = 3; end
      // if  test2 <= atol, istop = 2; end
      // if  test1 <= rtol, istop = 1; end
      if (test3 <= ctol) istop = 3;
      if (test2 <= atol) istop = 2;
      if (test1 <= rtol) istop = 1;

      // See if it is time to print something.

      // prnt = 0;
      // if n <= 40, prnt = 1; end
      // if itn <= 10, prnt = 1; end
      // if itn >= itnlim - 10, prnt = 1; end
      // if rem(itn, 10) == 0, prnt = 1; end
      // if test3 <= 2 * ctol, prnt = 1; end
      // if test2 <= 10 * atol, prnt = 1; end
      // if test1 <= 10 * rtol, prnt = 1; end
      // if istop ~= 0, prnt = 1; end
      bool prnt = false;
      if (A.num_cols <= 40) prnt = true;
      if (itn <= 10) prnt = true;
      if (itn >= itnlim - 10) prnt = true;
      if (itn % 10 == 0) prnt = true;
      if (test3 <= 2 * ctol) prnt = true;
      if (test2 <= 10 * atol) prnt = true;
      if (test1 <= 10 * rtol) prnt = true;
      if (istop != 0) prnt = true;

      // if prnt
      if (prnt)
      {
        // if show
        if (show)
        {
          // str1 = sprintf('%6g %12.5e', itn, x(1));
          std::string str1;
          {
            std::ostringstream sstream;
            sstream << std::scientific << std::setw(6) << itn << " " << std::setw(12) << std::setprecision(5) << x[1 - 1];
            str1 = sstream.str();
          }

          // str2 = sprintf(' %10.3e %10.3e', r1norm, r2norm);
          std::string str2;
          {
            std::ostringstream sstream;
            sstream << std::scientific << " " << std::setw(10) << std::setprecision(3) << r1norm << " " << std::setw(10) << std::setprecision(3) << r2norm;
            str2 = sstream.str();
          }

          // str3 = sprintf('  %8.1e %8.1e', test1, test2);
          std::string str3;
          {
            std::ostringstream sstream;
            sstream << std::scientific << "  " << std::setw(8) << std::setprecision(1) << test1 << " " << std::setw(8) << std::setprecision(1) << test2;
            str3 = sstream.str();
          }

          // str4 = sprintf(' %8.1e %8.1e', Anorm, Acond);
          std::string str4;
          {
            std::ostringstream sstream;
            sstream << std::scientific << " " << std::setw(8) << std::setprecision(1) << Anorm << " " << std::setw(8) << std::setprecision(1) << Acond;
            str4 = sstream.str();
          }

          // disp([str1 str2 str3 str4])
          LOG(INFO) << str1 << str2 << str3 << str4;
        }
        // end
      }
      // end

      // if istop > 0, break, end
      if (istop > 0) break;
    }
    // end
    // End of iteration loop.

    // Print the stopping condition.

    // if show
    if (show)
    {
      // disp(msg(istop + 1, :))
      LOG(INFO) << msg[istop];

      // disp(' ')
      LOG(INFO) << " ";

      // str1 = sprintf('istop =%8g   r1norm =%8.1e', istop, r1norm);
      std::string str1;
      {
        std::ostringstream sstream;
        sstream << std::scientific << "istop =" << std::setw(8) << istop << "   r1norm =" << std::setw(8) << std::setprecision(1) << r1norm;
        str1 = sstream.str();
      }

      // str2 = sprintf('Anorm =%8.1e   Arnorm =%8.1e', Anorm, Arnorm);
      std::string str2;
      {
        std::ostringstream sstream;
        sstream << std::scientific << "Anorm =" << std::setw(8) << std::setprecision(1) << Anorm << "   Arnorm =" << std::setw(8) << std::setprecision(1) << Arnorm;
        str2 = sstream.str();
      }

      // str3 = sprintf('itn   =%8g   r2norm =%8.1e', itn, r2norm);
      std::string str3;
      {
        std::ostringstream sstream;
        sstream << std::scientific << "itn   =" << std::setw(8) << itn << "   r2norm =" << std::setw(8) << std::setprecision(1) << r2norm;
        str3 = sstream.str();
      }

      // str4 = sprintf('Acond =%8.1e   xnorm  =%8.1e', Acond, xnorm);
      std::string str4;
      {
        std::ostringstream sstream;
        sstream << std::scientific << "Acond =" << std::setw(8) << std::setprecision(1) << Acond << "   xnorm  =" << std::setw(8) << std::setprecision(1) << xnorm;
        str4 = sstream.str();
      }

      // disp([str1 '   ' str2])
      LOG(INFO) << str1 << "   " << str2;

      // disp([str3 '   ' str4])
      LOG(INFO) << str3 << "   " << str4;

      // disp(' ')
      LOG(INFO) << " ";
    }
    // end
  }

private:
  const LinearOperator& A;
  const LinearOperator& At;
  const Array1d& b;
};

#endif//SparseLeastSquares_h