#ifndef CostFunction_hpp
#define CostFunction_hpp

template<typename JacobianMatrixType, typename VariableVectorType, typename ResidualVectorType>
class CostFunction
{
public:
	virtual void ComputeJacobian(const VariableVectorType& x, JacobianMatrixType& jacobian) const = 0;
	virtual void ComputeResidual(const VariableVectorType& x, ResidualVectorType& residual) const = 0;
	virtual ~CostFunction() { }
};

#endif//CostFunction_hpp