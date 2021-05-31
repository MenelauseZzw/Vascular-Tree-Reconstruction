#include "CostFunction.hpp"
#include "LSQR.hpp"
#include <boost/log/trivial.hpp>
#include <cusp/array1d.h>
#include <cusp/blas/blas.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>

template<typename JacobianMatrixType, typename VariableVectorType, typename ResidualVectorType, typename ValueType>
void LevenbergMarquardtMinimizer(
	const CostFunction<JacobianMatrixType, VariableVectorType, ResidualVectorType>& func,
	VariableVectorType& x, ValueType& damp, ValueType dampmin, ValueType tolx, ValueType tolf, ValueType tolg, int& itn, int itnlim,
	std::vector<ValueType>& costValue, double tau)
{
	typedef typename JacobianMatrixType::memory_space MemorySpace;
	typedef cusp::array1d<ValueType, MemorySpace> GradientVectorType;

	// Local constants
	const ValueType E = 2;
	const ValueType D = 0.5;

	const ValueType pi1 = 0.01;
	const ValueType pi2 = 0.75;

	const ValueType damplim = 32000;

	itn = 0;

	JacobianMatrixType jacobian;
	func.ComputeJacobian(x, jacobian);

	JacobianMatrixType jacobiant;

	GradientVectorType gradient(jacobian.num_cols);

	ResidualVectorType residualx(jacobian.num_rows);
	ResidualVectorType residualxpy(jacobian.num_rows);

	VariableVectorType y(jacobian.num_cols);
	VariableVectorType xpy(jacobian.num_cols);

	cusp::transpose(jacobian, jacobiant);

	func.ComputeResidual(x, residualx);

	ValueType normSqResidualx = cusp::blas::dot(residualx, residualx);

	ValueType intialValue = normSqResidualx;
	costValue.push_back(normSqResidualx);

	for (;;)
	{
		ValueType rho;
		ValueType normSqResidualxpy;

		{
			// Local constants
			const ValueType atol = 1e-3;
			const ValueType btol = 1e-3;
			const ValueType conlim = 0;
			const int itnlim = 4 * jacobian.num_cols;

			// Local variables
			ValueType Anorm, Acond, rnorm, Arnorm, xnorm;
			int istop, itn;

			LSQR(jacobian, jacobiant, residualx, damp, y, atol, btol, conlim, itnlim, istop, itn, Anorm, Acond, rnorm, Arnorm, xnorm);

			cusp::blas::axpby(x, y, xpy, 1, -1);

			func.ComputeResidual(xpy, residualxpy);

			normSqResidualxpy = cusp::blas::dot(residualxpy, residualxpy);
			rho = (normSqResidualx - normSqResidualxpy) / (normSqResidualx - rnorm * rnorm);
            costValue.push_back(normSqResidualxpy);

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
				BOOST_LOG_TRIVIAL(info) << "x-convergence criterion is signalled";
			}

			if ((normSqResidualx - normSqResidualxpy) / normSqResidualxpy <= tolf)
			{
				convergence = true;
				BOOST_LOG_TRIVIAL(info) << "Function convergence criterion is signalled";
			}

			cusp::multiply(jacobiant, residualx, gradient);

			if (cusp::blas::nrm2(gradient) <= tolg)
			{
				convergence = true;
				BOOST_LOG_TRIVIAL(info) << "Gradient convergence criterion is signalled";
			}

			if (convergence) break;

			//costValue.push_back(normSqResidualxpy);

			std::swap(residualxpy, residualx);
			normSqResidualx = normSqResidualxpy;

			func.ComputeJacobian(x, jacobian);
			cusp::transpose(jacobian, jacobiant);

			itn = itn + 1;

			if (rho > pi2) damp = D * damp;
			if (damp < dampmin) damp = 0;
			if (itn > itnlim) break;
		}

		if (damp > damplim) break;
	}
}
