#ifndef LinearCombination_hpp
#define LinearCombination_hpp

#include "CostFunction.hpp"
#include <type_traits>

template<int NumDimensions, typename ValueType, typename IndexType, typename MemorySpace>
class LinearCombination : public CostFunction <
	cusp::csr_matrix<IndexType, ValueType, MemorySpace>, cusp::array1d<ValueType, MemorySpace>, cusp::array1d<ValueType, MemorySpace> >
{
public:
	typedef cusp::csr_matrix<IndexType, ValueType, MemorySpace> JacobianMatrixType;
	typedef cusp::array1d<ValueType, MemorySpace> VariableVectorType;
	typedef cusp::array1d<ValueType, MemorySpace> ResidualVectorType;

	void ComputeJacobian(const VariableVectorType& x, JacobianMatrixType& jacobian) const;
	void ComputeResidual(const VariableVectorType& x, ResidualVectorType& residual) const;

	int VariableVectorLength() const { return variableVectorLength; }
	int ResidualVectorLength() const { return residualVectorLength1 + residualVectorLength2; }
	int JacobianVectorLength() const { return jacobianVectorLength1 + jacobianVectorLength2; }

protected:
	template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5, typename Vector6>
	LinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2, Vector6&& weights3, double voxelPhysicalSize, double tau)
		:
		tildeP(std::forward<Vector1>(tildeP)),
		indices1(std::forward<Vector2>(indices1)), indices2(std::forward<Vector3>(indices2)),
		weights1(std::forward<Vector4>(weights1)), weights2(std::forward<Vector5>(weights2)), weights3(std::forward<Vector6>(weights3)),
		residualVectorLength1(tildeP.size() / NumDimensions), residualVectorLength2(indices1.size()),
		jacobianVectorLength1(2 * tildeP.size()), jacobianVectorLength2(4 * NumDimensions * indices1.size()),
		variableVectorLength(2 * tildeP.size()),
		voxelPhysicalSize(voxelPhysicalSize),
		tau(tau)
	{
	}

	virtual void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength) const = 0;
	virtual void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength) const = 0;

	virtual void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength, double tau) const = 0;
	virtual void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength, double tau) const = 0;

private:
	typedef cusp::array1d<ValueType, MemorySpace> JacobianVectorType;

	void ComputeJacobian(const VariableVectorType& x, JacobianVectorType& jacobian) const;

	/*void SetUpColumnIndices(cusp::array1d<IndexType, MemorySpace>& columnIndices) const;*/
	void SetUpColumnIndices(cusp::array1d<IndexType, MemorySpace>& columnIndices) const;
	void SetUpRowPointers(cusp::array1d<IndexType, MemorySpace>& rowPointers) const;

	const cusp::array1d<ValueType, MemorySpace> tildeP;
	const cusp::array1d<IndexType, MemorySpace> indices1;
	const cusp::array1d<IndexType, MemorySpace> indices2;
	const cusp::array1d<ValueType, MemorySpace> weights1;
	const cusp::array1d<ValueType, MemorySpace> weights2;
	const cusp::array1d<ValueType, MemorySpace> weights3;

	const int residualVectorLength1;
	const int residualVectorLength2;
	const int jacobianVectorLength1;
	const int jacobianVectorLength2;
	const int variableVectorLength;
	const double voxelPhysicalSize;
	const double tau;
};

// CPU version does not support the oriented curvature and divergence prior
template<int NumDimensions, typename ValueType, typename IndexType = int>
class CpuLinearCombination : public LinearCombination < NumDimensions, ValueType, IndexType, cusp::host_memory >
{
public:
	typedef LinearCombination<NumDimensions, ValueType, IndexType, cusp::host_memory> Parent;
	typedef cusp::host_memory MemorySpace;

	template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5, typename Vector6>
	CpuLinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2, Vector6&& weights3, double voxelPhysicalSize, double tau)
		: Parent(tildeP, indices1, indices2, weights1, weights2, weights3, voxelPhysicalSize, tau)
	{
	}

	~CpuLinearCombination()
	{
	}

protected:
	void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength) const;
	void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength) const;

	void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength, double tau) const;
	void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength, double tau) const;
};

template<int NumDimensions, typename ValueType, typename IndexType = int>
class GpuLinearCombination : public LinearCombination < NumDimensions, ValueType, IndexType, cusp::device_memory >
{
public:
	typedef LinearCombination<NumDimensions, ValueType, IndexType, cusp::device_memory> Parent;
	typedef cusp::device_memory MemorySpace;

	template<typename Vector1, typename Vector2, typename Vector3, typename Vector4, typename Vector5, typename Vector6>
	GpuLinearCombination(Vector1&& tildeP, Vector2&& indices1, Vector3&& indices2, Vector4&& weights1, Vector5&& weights2, Vector6&& weights3, double voxelPhysicalSize, double tau)
		: Parent(tildeP, indices1, indices2, weights1, weights2, weights3, voxelPhysicalSize, tau)
	{
	}

	~GpuLinearCombination()
	{
	}

protected:
	void ComputeResidual1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pResidual, double voxelPhysicalSize, int residualVectorLength) const;
	void ComputeJacobian1(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, ValueType* pJacobian, double voxelPhysicalSize, int residualVectorLength) const;

	void ComputeResidual2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pResidual, int residualVectorLength, double tau) const;
	void ComputeJacobian2(const ValueType* pTildeP, const ValueType* pS, const ValueType* pT, const ValueType* pWeights, const ValueType* pWeights3, const IndexType* pIndices1, const IndexType* pIndices2, ValueType* pJacobian, int residualVectorLength, double tau) const;
};

#include "LinearCombination.inl"

#endif//LinearCombination_hpp
