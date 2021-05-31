// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "MRFEnergy.h"
#include "typeBinary.h"
#include "instances.h"
#include "MRFEnergy.cpp"
#include "minimize.cpp"
#include "ordering.cpp"
#include "treeProbabilities.cpp"
#include "FileReader.hpp"
#include "FileWriter.hpp"
#include "DoLevenbergMarquardtMinimizer.hpp"
#include "DoProjectionOntoLine.hpp"
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <vector>
#include <numeric>
#include "math.h"

template<typename T, int Dimensions>
T dotproduct(const std::vector<T>& A, const std::vector<T>& B)
{
	T sum = 0;
	for (int i = 0; i < Dimensions; i++) {
		sum += A[i] * B[i];
	}
	return sum;
}

template<int NumDimensions, typename ValueType, typename IndexType>
void DoTRWSMinimizer(
	std::vector<ValueType>& points,
	std::vector<ValueType>& tangentLinesPoints1,
	std::vector<ValueType>& tangentLinesPoints2,
	std::vector<ValueType>& radiuses,
	std::vector<IndexType>& indices1,
	std::vector<IndexType>& indices2,
	const ValueType& maxRad,
	const ValueType& minRad,
	const ValueType& alpha,
	const ValueType& beta,
	const ValueType& gamma,
	const IndexType& rootIndice,
	std::vector<ValueType>& gap,
	double tau)
{
	// TRW-S on binary energy E(x)
	MRFEnergy<TypeBinary>* mrf;
	MRFEnergy<TypeBinary>::NodeId* nodes;
	MRFEnergy<TypeBinary>::Options options;
	TypeBinary::REAL energy, lowerBound;

	const int nodeNum = points.size() / 3;
	int* x;
	x = (int*)malloc(nodeNum * sizeof(int));

	mrf = new MRFEnergy<TypeBinary>(TypeBinary::GlobalSize());
	nodes = new MRFEnergy<TypeBinary>::NodeId[nodeNum];

	for (int n = 0; n < nodeNum; n++)
	{
		if (n == rootIndice) {
			nodes[n] = mrf->AddNode(TypeBinary::LocalSize(), TypeBinary::NodeData(0, 0));
			//nodes[n] = mrf->AddNode(TypeBinary::LocalSize(), TypeBinary::NodeData(0, 5)); //unary potential for the root node
		}
		else {
			nodes[n] = mrf->AddNode(TypeBinary::LocalSize(), TypeBinary::NodeData(0, 0));
		}
	}

	ValueType threshold = tau;
	ValueType weightCurv = alpha * alpha; /// 2;
	ValueType weightDiv = beta * beta;
	ValueType weightRad = gamma * gamma;

	// get edge data
	// here edge data includes two parts consisting of "curvature" and "radius"
	auto EdgeData = [&](const int ip, const int iq, const int xp, const int xq, bool Flag = true)
	{
		ValueType InProd;
		ValueType res;
		std::vector<ValueType> baseLine;
		std::vector<ValueType> Lp;
		std::vector<ValueType> Lq;

		for (int i = 0; i < NumDimensions; i++)
		{
			baseLine.push_back(points[3 * ip + i] - points[3 * iq + i]);
			Lp.push_back((-2 * xp + 1)*tangentLinesPoints2[3 * ip + i] - (-2 * xp + 1)*tangentLinesPoints1[3 * ip + i]);
			Lq.push_back((-2 * xq + 1)*tangentLinesPoints2[3 * iq + i] - (-2 * xq + 1)*tangentLinesPoints1[3 * iq + i]);
		}
		InProd = dotproduct<ValueType, 3>(Lp, Lq) / (sqrt(dotproduct<ValueType, 3>(Lp, Lp)+1e-15)*sqrt(dotproduct<ValueType, 3>(Lq, Lq)+1e-15));
		if (Flag)
		{
			// curvature part
			if (InProd >= threshold)
			{
				ValueType SquareDistPtoLq = dotproduct<ValueType, 3>(baseLine, baseLine) - (dotproduct<ValueType, 3>(baseLine, Lq)*dotproduct<ValueType, 3>(baseLine, Lq)) / (dotproduct<ValueType, 3>(Lq, Lq)+1e-15);
				ValueType SquareDistQtoLp = dotproduct<ValueType, 3>(baseLine, baseLine) - (dotproduct<ValueType, 3>(baseLine, Lp)*dotproduct<ValueType, 3>(baseLine, Lp)) / (dotproduct<ValueType, 3>(Lp, Lp)+1e-15);
				
                // need to change according to which version of curvature to be used
				//res = weightCurv * (sqrt(SquareDistPtoLq+1e-15) + sqrt(SquareDistQtoLp+1e-15)) / sqrt(dotproduct<ValueType, 3>(baseLine, baseLine)+1e-15);
                res = weightCurv * sqrt(SquareDistQtoLp+1e-15) / sqrt(dotproduct<ValueType, 3>(baseLine, baseLine)+1e-15);
                
				ValueType maxRadius = std::max(radiuses[ip], radiuses[iq]);
				maxRadius = (maxRadius - minRad) / (maxRad - minRad); //normalization into [0,1]
																	  //res = res * exp(-beta*maxRadius); // "discount" from radius on curvature part
																	  //std::cout << "maxradius:" << exp(-maxRadius) << "   ";
			}
			else
			{
				res = weightCurv * 1.1; //2.0; //(sinA^2 + sinB^2)*exp(-beta*maxRad) <= (1- cos(A+B)cos(A-B))*1.0 <= 2.0
			}

			// divergence part or collision part
			ValueType LpNorm = sqrt(dotproduct<ValueType, 3>(Lp, Lp)+1e-15);
			ValueType LqNorm = sqrt(dotproduct<ValueType, 3>(Lq, Lq)+1e-15);
			ValueType divergence = (dotproduct<ValueType, 3>(Lp, baseLine) / LpNorm - dotproduct<ValueType, 3>(Lq, baseLine) / LqNorm);
			ValueType baseNorm = sqrt(dotproduct<ValueType, 3>(baseLine, baseLine) + 1e-15);
			if (divergence < -0.20*baseNorm)
			{
				res += weightDiv * (-divergence);
			}

			// radius part
			//std::cout << "(radip,radiq):" << radiuses[ip] << "," << radiuses[iq] << "  ";
			//ValueType radiq = (radiuses[iq] - minRad) / (maxRad - minRad); // normalization
			//ValueType radip = (radiuses[ip] - minRad) / (maxRad - minRad);
			ValueType radiq = radiuses[iq];
			ValueType radip = radiuses[ip];

			if (radiuses[ip] < radiuses[iq])
			{
				for (int j = 0; j < NumDimensions; j++)
				{
					baseLine[j] = -1 * baseLine[j];
				}
				ValueType radPenalty1 = dotproduct<ValueType, 3>(Lp, baseLine);
				ValueType radPenalty2 = dotproduct<ValueType, 3>(Lq, baseLine);

				// bigger than 0 indicates that the orientation points from thin vessel to thick vessel
				if ((radPenalty1 > 0) && (radPenalty2 > 0)) // two tangents are aligned
				{
					res += weightRad * (radiq*radiq - radip*radip);
				}
			}
			else
			{
				ValueType radPenalty1 = dotproduct<ValueType, 3>(Lp, baseLine);
				ValueType radPenalty2 = dotproduct<ValueType, 3>(Lq, baseLine);

				// bigger than 0 indicates that the orientation points from thin vessel to thick vessel
				if ((radPenalty1 > 0) && (radPenalty2 > 0)) // two tangents are aligned
				{
					res += weightRad * (radip*radip - radiq*radiq);
				}
			}
			return res;
		}
		else
		{
			return InProd;
		}
	};

	for (int e = 0; e < indices1.size(); e++)
	{
		IndexType n1 = indices1[e];
		IndexType n2 = indices2[e];
		mrf->AddEdge(nodes[n1], nodes[n2], TypeBinary::EdgeData(EdgeData(n1, n2, 0, 0), EdgeData(n1, n2, 0, 1), EdgeData(n1, n2, 1, 0), EdgeData(n1, n2, 1, 1)));
	}

	// compute the initial energy
	ValueType SumEn = 0;
	for (int e = 0; e < indices1.size(); e++)
	{
		IndexType n1 = indices1[e];
		IndexType n2 = indices2[e];
		ValueType res = EdgeData(n1, n2, 0, 0);
		SumEn += res;
	}
	std::cout << "inital energy: " << SumEn << std::endl;
	mrf->SetAutomaticOrdering();

	//TRW-S algorithm
	options.m_iterMax = 1000;//2000;
	mrf->Minimize_TRW_S(options, lowerBound, energy);

	//read solutions
	for (int n = 0; n < nodeNum; n++)
	{
		x[n] = mrf->GetSolution(nodes[n]);
	}

	SumEn = 0;
	for (int e = 0; e < indices1.size(); e++)
	{
		IndexType n1 = indices1[e];
		IndexType n2 = indices2[e];
		ValueType res = EdgeData(n1, n2, x[n1], x[n2]);
		SumEn += res;
	}
	gap.push_back(energy - lowerBound);

	for (int i = 0; i < points.size() / 3; i++)
	{
		if (x[i] == 1)
		{
			for (int j = 0; j < NumDimensions; j++)
			{
				ValueType temp = tangentLinesPoints1[3 * i + j];
				tangentLinesPoints1[3 * i + j] = tangentLinesPoints2[3 * i + j];
				tangentLinesPoints2[3 * i + j] = temp;
			}
		}
	}

	/*std::vector<IndexType> newIndices1;
	std::vector<IndexType> newIndices2;

    // orientated curvature based neighborhood connectivity
	// used for neighborhood setup once at the beginning
	for (int e = 0; e < indices1.size(); e++)
	{
		IndexType n1 = indices1[e];
		IndexType n2 = indices2[e];
		ValueType res = EdgeData(n1, n2, 0, 0, false);
		if (res > threshold)
		{
			newIndices1.push_back(n1);
			newIndices2.push_back(n2);
		}
		else
		{
			consterm += 4;
		}
	}

	indices1 = newIndices1;
	indices2 = newIndices2;*/
}

void DoLevenbergMarquardtMinimizer(
	const std::string& inputFileName,
	const std::string& outputFileName,
	double lambda,
	double voxelPhysicalSize,
	int maxNumberOfIterations,
	int gpuDevice,
	int rootIndice,
	double beta,
	double gamma,
	double tau)
{
	const unsigned int NumDimensions = 3;

	typedef double ValueType;
	typedef int IndexType;

	const std::string measurementsDataSetName = "measurements";
	const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
	const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
	const std::string radiusesDataSetName = "radiuses";
	const std::string objectnessMeasureDataSetName = "objectnessMeasure";
	const std::string positionsDataSetName = "positions";

	const std::string indices1DataSetName = "indices1";
	const std::string indices2DataSetName = "indices2";

	const std::string costValueName = "costValue";

	FileReader inputFileReader(inputFileName);

	std::vector<ValueType> measurements;
	std::vector<ValueType> tangentLinesPoints1;
	std::vector<ValueType> tangentLinesPoints2;
	std::vector<ValueType> radiuses;
	std::vector<ValueType> objectnessMeasure;

	std::vector<IndexType> indices1;
	std::vector<IndexType> indices2;

	std::vector<ValueType> gap;
	//std::vector<ValueType> positions;
	std::vector<ValueType> costValue;

	//inputFileReader.Read(positionsDataSetName, positions);

	inputFileReader.Read(measurementsDataSetName, measurements);
	inputFileReader.Read(tangentLinesPoints1DataSetName, tangentLinesPoints1);
	inputFileReader.Read(tangentLinesPoints2DataSetName, tangentLinesPoints2);
	inputFileReader.Read(radiusesDataSetName, radiuses);
	inputFileReader.Read(objectnessMeasureDataSetName, objectnessMeasure);

	inputFileReader.Read(indices1DataSetName, indices1);
	inputFileReader.Read(indices2DataSetName, indices2);

	ValueType maxRad = *std::max_element(radiuses.begin(), radiuses.end());
	ValueType minRad = *std::min_element(radiuses.begin(), radiuses.end());

	BOOST_LOG_TRIVIAL(info) << "measurements.size = " << measurements.size();
	BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints1.size = " << tangentLinesPoints1.size();
	BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints2.size = " << tangentLinesPoints2.size();
	BOOST_LOG_TRIVIAL(info) << "radiuses.size = " << radiuses.size();
	BOOST_LOG_TRIVIAL(info) << "objectnessMeasure.size = " << objectnessMeasure.size();

	BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
	BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();
	
	std::vector<ValueType> lambdas(indices1.size(), lambda);
	std::vector<ValueType> betas(indices1.size(), beta);

	//Coordinate Descent method(alternate between Levenbergmarquardt and TRWS)
	DoTRWSMinimizer<NumDimensions, ValueType, IndexType>(measurements, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, maxRad, minRad, lambda, beta, gamma, rootIndice, gap, tau);

	int iter = 0;
	while (iter < 3)
	{
		if (gpuDevice != -1)
		{
			if (cudaErrorInvalidDevice == cudaSetDevice(gpuDevice))
				cudaGetDevice(&gpuDevice);

			cudaDeviceProp gpuDeviceProp;
			cudaGetDeviceProperties(&gpuDeviceProp, gpuDevice);

			BOOST_LOG_TRIVIAL(info) << "Device " << gpuDeviceProp.name << "(" << gpuDevice << ")";
			BOOST_LOG_TRIVIAL(info) << "Maximum size of each dimension of a block (" << gpuDeviceProp.maxThreadsDim[0] << " " << gpuDeviceProp.maxThreadsDim[1] << " " << gpuDeviceProp.maxThreadsDim[2] << ") ";
			BOOST_LOG_TRIVIAL(info) << "Maximum number of threads per block " << gpuDeviceProp.maxThreadsPerBlock;
			BOOST_LOG_TRIVIAL(info) << "Maximum resident threads per multiprocessor " << gpuDeviceProp.maxThreadsPerMultiProcessor;

			DoGpuLevenbergMarquardtMinimizer<NumDimensions>(
				measurements,
				tangentLinesPoints1,
				tangentLinesPoints2,
				radiuses,
				indices1,
				indices2,
				lambdas,
				betas,
				maxNumberOfIterations,
				voxelPhysicalSize,
				costValue,
				tau);
		}
		else
		{
			DoCpuLevenbergMarquardtMinimizer<NumDimensions>(
				measurements,
				tangentLinesPoints1,
				tangentLinesPoints2,
				radiuses,
				indices1,
				indices2,
				lambdas,
				betas,
				maxNumberOfIterations,
				voxelPhysicalSize,
				costValue,
				tau);
		}

		std::vector<ValueType> positions;

		DoCpuProjectionOntoLine<NumDimensions>(
			measurements,
			tangentLinesPoints1,
			tangentLinesPoints2,
			positions);

		iter++;

		DoTRWSMinimizer<NumDimensions, ValueType, IndexType>(positions, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, maxRad, minRad, lambda, beta, gamma, rootIndice, gap, tau);

	}

	std::vector<ValueType> positions;

	DoCpuProjectionOntoLine<NumDimensions>(
		measurements,
		tangentLinesPoints1,
		tangentLinesPoints2,
		positions);

	//write to outputfile
	FileWriter outputFileWriter(outputFileName);

	outputFileWriter.Write(measurementsDataSetName, measurements);
	outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
	outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
	outputFileWriter.Write(radiusesDataSetName, radiuses);
	outputFileWriter.Write(objectnessMeasureDataSetName, objectnessMeasure);
	outputFileWriter.Write(positionsDataSetName, positions);

	outputFileWriter.Write(indices1DataSetName, indices1);
	outputFileWriter.Write(indices2DataSetName, indices2);

	std::string gapname = "gap";
	outputFileWriter.Write(gapname, gap);

	outputFileWriter.Write(costValueName, costValue);
}

int main(int argc, char *argv[])
{
	namespace po = boost::program_options;

	int maxNumberOfIterations = 1000;
	double lambda;
	double voxelPhysicalSize;
	std::string inputFileName;
	std::string outputFileName;
	int gpuDevice = -1;
	int rootIndice = 0;
	double beta;
	double gamma = 0;
	double tau;

	po::options_description desc;
	desc.add_options()
		("help", "print usage message")
		("lambda", po::value(&lambda)->required(), "the value of regularization parameter/ weight of curvature potential")
		("voxelPhysicalSize", po::value(&voxelPhysicalSize)->required(), "the physical size of a voxel")
		("gpuDevice", po::value(&gpuDevice), "gpu device should be used; otherwise, run on cpu")
		("maxNumberOfIterations", po::value(&maxNumberOfIterations), "the upper limit on the number of iterations")
		("inputFileName", po::value(&inputFileName)->required(), "the name of the input file")
		("outputFileName", po::value(&outputFileName)->required(), "the name of the output file")
		("rootIndice", po::value(&rootIndice), "the indice of root of the vessel tree")
		("beta", po::value(&beta)->required(), "the weight of divergence potential")
		("gamma", po::value(&gamma), "the weight of radiuses potential")
		("tau", po::value(&tau)->required(), "the internal hyperparameter of the oriented curvature");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		desc.print(std::cout);
		return EXIT_SUCCESS;
	}

	try
	{
		DoLevenbergMarquardtMinimizer(inputFileName, outputFileName, lambda, voxelPhysicalSize, maxNumberOfIterations, gpuDevice, rootIndice, beta, gamma, tau);
		return EXIT_SUCCESS;
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}
