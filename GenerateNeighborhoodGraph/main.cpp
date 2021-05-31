// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
#include "FileWriter.hpp"
#include <algorithm>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <functional>
#include <itkContinuousIndex.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIndex.h>
#include <itkPoint.h>
#include <itkSize.h>
#include <itkShapedNeighborhoodIterator.h>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <flann/flann.hpp>
#include <iostream>

template<typename ValueType, typename IndexType>
void GenerateAKnnGraph(
	const std::vector<ValueType>& positions,
	const std::vector<ValueType>& tangentLinesPoints1,
	const std::vector<ValueType>& tangentLinesPoints2,
	std::vector<IndexType>& indices1,
	std::vector<IndexType>& indices2,
	int numDimensions,
	double aspectRatio,
	int knnValue,
	bool mutualLink,
	double distConstraint,
	ValueType voxelSize)
{
	using namespace flann;
	using namespace std;

	int knn = 200;

	vector<IndexType> indices1_;
	vector<IndexType> indices2_;

	const Matrix<ValueType> dataset(const_cast<ValueType*>(positions.data()), positions.size() / numDimensions, numDimensions);
	
	//ValueType voxelSize = 1.0;//0.046;

    Eigen::Matrix3d A;
	A << 0.5*voxelSize, 0, 0,
		 0, 0.5*aspectRatio*voxelSize, 0,
		 0, 0, 0.5*aspectRatio*voxelSize;
	std::cout << A<<endl;

	Index<L2<ValueType>> index(dataset, flann::KDTreeIndexParams(1));
	index.buildIndex();

	vector<IndexType> indicesData(dataset.rows * (knn + 1));
	Matrix<IndexType> indices(&indicesData[0], dataset.rows, knn + 1);

	vector<ValueType> distancesData(dataset.rows * (knn + 1));
	Matrix<ValueType> distances(&distancesData[0], dataset.rows, knn + 1);

	index.knnSearch(dataset, indices, distances, knn + 1, flann::SearchParams(-1));

	indices1.reserve(knnValue * positions.size());
	indices2.reserve(indices1.size());

	vector<IndexType> indexSearch;
	indexSearch.push_back(0);

	ValueType ratio = 0;

	for (IndexType index1 = 0; index1 != dataset.rows; ++index1)
	{
		vector<IndexType> tempIndex2;

		for (IndexType i = 0; i != knn + 1; ++i)
		{
			const IndexType index2 = indices[index1][i];
			if (index1 == index2) continue;
			if (index2 == -1) break;
			
			tempIndex2.push_back(index2);
		}

		const ValueType a1 = tangentLinesPoints2[numDimensions*index1 + 0] - tangentLinesPoints1[numDimensions*index1 + 0];
		const ValueType b1 = tangentLinesPoints2[numDimensions*index1 + 1] - tangentLinesPoints1[numDimensions*index1 + 1];
		const ValueType c1 = tangentLinesPoints2[numDimensions*index1 + 2] - tangentLinesPoints1[numDimensions*index1 + 2];
		ValueType norm1 = sqrt(a1*a1 + b1*b1 + c1*c1);

		ValueType a2, b2, c2;
		if (a1 != 0)
		{
			b2 = 1;
			c2 = 1;
			a2 = (-b1 - c1) / a1;
		}
		else if (b1 != 0)
		{
			a2 = 1;
			c2 = 1;
			b2 = (-a1 - c1) / b1;
		}
		else
		{
			a2 = 1;
			b2 = 1;
			c2 = (-a1 - b1) / c1;
		}
		ValueType norm2 = sqrt(a2*a2 + b2*b2 + c2*c2);

		ValueType a3, b3, c3;
		a3 = b1*c2 - c1*b2;
		b3 = c1*a2 - a1*c2;
		c3 = a1*b2 - b1*a2;
		ValueType norm3 = sqrt(a3*a3 + b3*b3 + c3*c3);

		Eigen::Matrix3d R;
		R << a1/norm1, a2/norm2, a3/norm3,
			b1/norm1, b2/norm2, b3/norm3,
			c1/norm1, c2/norm2, c3/norm3;

		Eigen::Matrix3d S_ = R*A;
		Eigen::Matrix3d S =S_* R.transpose();

		Eigen::LLT<Eigen::Matrix3d> lltOfS(S);
		Eigen::Matrix3d L = lltOfS.matrixL();
		Eigen::Matrix3d Lt = L.transpose();

		vector<ValueType> knnPositions;// construction of new KNN search for aKNN graph

	    Eigen::Map<const Eigen::RowVector3d> v1_(positions.data() + numDimensions*index1, 1, numDimensions);
		Eigen::Vector3d v1 = Lt*v1_.transpose();
		knnPositions.push_back(v1(0));
		knnPositions.push_back(v1(1));
		knnPositions.push_back(v1(2));

		vector<IndexType> tempIndex2_;

		for (IndexType i = 0; i != knn; i++)
		{
			IndexType ind2 = tempIndex2[i];
			Eigen::Map<const Eigen::RowVector3d> v2_(positions.data() + numDimensions*ind2, 1, numDimensions);
			Eigen::Vector3d v_ = (v2_ - v1_).transpose();
			double res_ = v_.dot(v_);

			//if (sqrt(res_+1e-15) > 3.5 * voxelSize)
				//continue; // constrained to grid neighborhood with radius of 3.5

			Eigen::Vector3d v2 = Lt*v2_.transpose();
			Eigen::Vector3d v = v2 - v1;

			double res = v.dot(v);
			double r_short = sqrt(aspectRatio * distConstraint * distConstraint * voxelSize / 2);
			
			if (sqrt(res + 1e-15) > r_short*voxelSize) //anisotropic nearest neighborhood radius
				continue; 

			tempIndex2_.push_back(ind2);

			knnPositions.push_back(v2(0));
			knnPositions.push_back(v2(1));
			knnPositions.push_back(v2(2));
		}

		//KNN search for knnValue aKNN graph
		int NNA = knnValue;
		int count = 0;

		ValueType max_rknn;

		if (tempIndex2_.size() <= NNA)
		{
			for (IndexType i = 0; i != tempIndex2_.size() ; ++i)
			{
				indices1_.push_back(index1);
				indices2_.push_back(tempIndex2_[i]);
				count++;
			}
		}
		else {
			const Matrix<ValueType> datasetANN(const_cast<ValueType*>(knnPositions.data()), knnPositions.size() / numDimensions, numDimensions);

			Index<L2<ValueType>> indexANN(datasetANN, flann::KDTreeIndexParams(1));
			indexANN.buildIndex();

			vector<IndexType> indicesDataANN(datasetANN.rows * (NNA + 1));
			Matrix<IndexType> indicesANN(&indicesDataANN[0], datasetANN.rows, NNA + 1);

			vector<ValueType> distancesDataANN(datasetANN.rows * (NNA + 1));
			Matrix<ValueType> distancesANN(&distancesDataANN[0], datasetANN.rows, NNA + 1);

			indexANN.knnSearch(datasetANN, indicesANN, distancesANN, NNA + 1, flann::SearchParams(-1));

			for (IndexType i = 0; i != NNA + 1; ++i)
			{
				const IndexType temp = indicesANN[0][i];
				if (0 == temp) continue;
				if (temp == -1) break;

				indices1_.push_back(index1);
				indices2_.push_back(tempIndex2_[temp-1]);
				count++;
			}
			//system("read -p 'Press Enter to continue...' var");

			//auto const& rknn = reinterpret_cast<const ValueType(&)[NNA + 1]>(distancesDataANN[index1*(NNA + 1)]);
			//max_rknn = *max_element(rknn, rknn + NNA + 1);
		}
		indexSearch.push_back(indexSearch[index1] + count);

		//Test for analyzing the radius ratio of 500NN and knn
		//auto const& r500 = reinterpret_cast<const ValueType(&)[knn + 1]>(distancesData[index1*(knn + 1)]);
		//auto max_r500 = *max_element(r500, r500 + knn + 1);
		
		//ratio = ratio + (max_r500 / (max_rknn + 1e-15));
		//cout << max_r500 << endl << max_rknn << endl;
		//cout << ratio << endl;
		//system("read -p 'Press Enter to continue...' var");
	}
	
	//ratio = ratio / dataset.rows;
	//cout << ratio << endl;


	//enforce mutual connection
	for (IndexType i = 0; i != indices1_.size(); ++i)
	{
		IndexType index1 = indices1_[i];
		IndexType index2 = indices2_[i];
		if (index2 == indexSearch.size() - 1)
		{
			for (IndexType j = indexSearch[index2]; j != indices2_.size(); ++j)
			{
				if (indices2_[j] == index1 || (!mutualLink))
				{
                    Eigen::Map<const Eigen::RowVector3d> vInd1(positions.data() + numDimensions*index1, 1, numDimensions);
                    Eigen::Map<const Eigen::RowVector3d> vInd2(positions.data() + numDimensions*index2, 1, numDimensions);
			        Eigen::Vector3d vInd = (vInd2 - vInd1).transpose();
			        double resInd = vInd.dot(vInd);

			        //if (sqrt(resInd+1e-15) <= 3.5 * voxelSize)
                       {
					     indices1.push_back(index1);
					     indices2.push_back(index2);
                       }
				}
			}
		}
		else
		{
			for (IndexType j = indexSearch[index2]; j != indexSearch[index2 + 1]; ++j)
			{
				if (indices2_[j] == index1 || (!mutualLink))
				{
					Eigen::Map<const Eigen::RowVector3d> vInd1(positions.data() + numDimensions*index1, 1, numDimensions);
                    Eigen::Map<const Eigen::RowVector3d> vInd2(positions.data() + numDimensions*index2, 1, numDimensions);
			        Eigen::Vector3d vInd = (vInd2 - vInd1).transpose();
			        double resInd = vInd.dot(vInd);

			        //if (sqrt(resInd+1e-15) <= 3.5 * voxelSize)
                       {
					     indices1.push_back(index1);
					     indices2.push_back(index2);
                       }
				}
			}
		}
	}
}

void DoGenerateNeighborhoodGraph(
  const std::string& inputFileName,
  const std::string& outputFileName,
  float thresholdBelow,
  bool aKnnGraph,
  double aspectRatio,
  int knnValue,
  bool mutualLink,
  double distConstraint,
  double voxelSize)
{
  using namespace boost;

  typedef float InputValueType;
  typedef double OutputValueType;
  typedef int OutputIndexType;

  constexpr unsigned int NumDimensions = 3;
  constexpr unsigned int VectorDimension = 1 + NumDimensions + 1; // each value consists of measure(1), eigenVector(n) and scale(1)

  constexpr unsigned int ObjectnessMeasureValueComponentIndex = 0;
  constexpr unsigned int ObjectnessMeasureTangentsComponentIndex = 1;
  constexpr unsigned int SigmaValueComponentIndex = 1 + NumDimensions;

  typedef itk::Vector<InputValueType, VectorDimension> PixelType;
  typedef itk::Image<PixelType, NumDimensions> ObjectnessMeasureImageType;

  typedef itk::Size<NumDimensions> SizeType;
  typedef itk::Index<NumDimensions> IndexType;
  typedef itk::ContinuousIndex<OutputValueType, NumDimensions> ContinuousIndexType;
  typedef itk::Point<OutputValueType, NumDimensions> PointType;

  typedef typename SizeType::SizeValueType SizeValueType;

  typedef itk::ShapedNeighborhoodIterator<ObjectnessMeasureImageType> ImageRegionIteratorType;
  typedef typename ImageRegionIteratorType::ConstIterator NeighborhoodIteratorType;

  typedef itk::ImageFileReader<ObjectnessMeasureImageType> FileReaderType;

  FileReaderType::Pointer imageReader =
    FileReaderType::New();

  imageReader->SetFileName(inputFileName);
  imageReader->Update();

  ObjectnessMeasureImageType::Pointer inputImage =
    imageReader->GetOutput();

  constexpr SizeValueType radiusOfNeighborhood = 1;

  SizeType neighborhoodRadius;
  neighborhoodRadius.Fill(radiusOfNeighborhood);

  ImageRegionIteratorType imageRegionIterator(neighborhoodRadius, inputImage, inputImage->GetLargestPossibleRegion());

  SizeValueType sizeOfNeighborhood = 1;

  for (unsigned int k = 0; k < NumDimensions; ++k)
  {
    sizeOfNeighborhood *= (radiusOfNeighborhood + 1 + radiusOfNeighborhood);
  }

  /*for (int i=-1; i<=1;i++)
	  for(int j=-1;j<=1;j++)
		  for (int k = -1; k <= 1; k++)
		  {
			  ImageRegionIteratorType::OffsetType off;

			  double dis = std::sqrt(i*i + j*j + k*k);
			  if (dis <= std::sqrt(3))
			  {
				  off[0] = static_cast<int>(i);
				  off[1] = static_cast<int>(j);
				  off[2] = static_cast<int>(k);
				  imageRegionIterator.ActivateOffset(off);
			  }
		  }*/
  for (SizeValueType i = 0; i < sizeOfNeighborhood; ++i)
  {
    imageRegionIterator.ActivateOffset(imageRegionIterator.GetOffset(i));
  }

  imageRegionIterator.DeactivateOffset(imageRegionIterator.GetOffset(imageRegionIterator.GetCenterNeighborhoodIndex()));

  typedef std::map<IndexType, OutputIndexType, typename IndexType::LexicographicCompare> IndexToOutputIndexMap;
  typedef IndexToOutputIndexMap::iterator IndexToOutputIndexMapIterator;
  IndexToOutputIndexMap indexToOutputIndex;

  std::vector<OutputValueType> measurements;
  std::vector<OutputValueType> tangentLinesPoints1;
  std::vector<OutputValueType> tangentLinesPoints2;
  std::vector<OutputValueType> radiuses;
  std::vector<OutputValueType> objectnessMeasure;

  std::vector<OutputIndexType> indices1;
  std::vector<OutputIndexType> indices2;

  std::function<OutputIndexType(IndexType const&, PixelType const&)> getIndexOrAdd = [&](IndexType const& indexOfPixel, PixelType const& valueAtPixel)
  {
    IndexToOutputIndexMapIterator iteratorAtPixel = indexToOutputIndex.find(indexOfPixel);
    OutputIndexType index;

    if (iteratorAtPixel != indexToOutputIndex.end())
    {
      index = iteratorAtPixel->second;
    }
    else
    {
      index = objectnessMeasure.size();
      const OutputValueType objectnessMeasureValue = valueAtPixel.GetElement(ObjectnessMeasureValueComponentIndex);
      const OutputValueType sigmaValue = valueAtPixel.GetElement(SigmaValueComponentIndex);

      objectnessMeasure.push_back(objectnessMeasureValue);

      PointType pointAtPixel;
      inputImage->TransformIndexToPhysicalPoint(indexOfPixel, pointAtPixel);

      ContinuousIndexType indexOfTangentLinePoint1;
      ContinuousIndexType indexOfTangentLinePoint2;

      for (unsigned int k = 0; k < NumDimensions; ++k)
      {
        const OutputValueType objectnessMeasureTangent = valueAtPixel.GetElement(ObjectnessMeasureTangentsComponentIndex + k);

        indexOfTangentLinePoint1.SetElement(k, indexOfPixel.GetElement(k) - objectnessMeasureTangent);
        indexOfTangentLinePoint2.SetElement(k, indexOfPixel.GetElement(k) + objectnessMeasureTangent);
      }

      PointType tangentLinePoint1;
      inputImage->TransformContinuousIndexToPhysicalPoint(indexOfTangentLinePoint1, tangentLinePoint1);

      PointType tangentLinePoint2;
      inputImage->TransformContinuousIndexToPhysicalPoint(indexOfTangentLinePoint2, tangentLinePoint2);

      copy(pointAtPixel.GetDataPointer(), pointAtPixel.GetDataPointer() + NumDimensions, std::back_inserter(measurements));
      copy(tangentLinePoint1.GetDataPointer(), tangentLinePoint1.GetDataPointer() + NumDimensions, std::back_inserter(tangentLinesPoints1));
      copy(tangentLinePoint2.GetDataPointer(), tangentLinePoint2.GetDataPointer() + NumDimensions, std::back_inserter(tangentLinesPoints2));

      radiuses.push_back(sigmaValue);

      indexToOutputIndex.insert({ indexOfPixel, index });
    }

    return index;
  };

  for (imageRegionIterator.GoToBegin(); !imageRegionIterator.IsAtEnd(); ++imageRegionIterator)
  {
    const PixelType* pValueAtCenterPixel = imageRegionIterator.GetCenterPointer();
    const InputValueType objectnessMeasureValueAtCenter = pValueAtCenterPixel->GetElement(ObjectnessMeasureValueComponentIndex);

    if (objectnessMeasureValueAtCenter < thresholdBelow) continue;
    const IndexType indexOfCenterPixel = imageRegionIterator.GetIndex();
    OutputIndexType outputIndexOfCenterPixel;

    outputIndexOfCenterPixel = getIndexOrAdd(indexOfCenterPixel, *pValueAtCenterPixel);

    for (NeighborhoodIteratorType neighborhoodIterator = imageRegionIterator.Begin(); !neighborhoodIterator.IsAtEnd(); ++neighborhoodIterator)
    {
      const IndexType indexOfNeighbor = indexOfCenterPixel + neighborhoodIterator.GetNeighborhoodOffset();

      if (!inputImage->GetLargestPossibleRegion().IsInside(indexOfNeighbor))
        continue;

      const PixelType valueAtNeighbor = neighborhoodIterator.Get();
      const InputValueType objectnessMeasureValueAtNeighbor = valueAtNeighbor.GetElement(ObjectnessMeasureValueComponentIndex);

      if (objectnessMeasureValueAtNeighbor < thresholdBelow) continue;

      OutputIndexType outputIndexOfNeighbor = getIndexOrAdd(indexOfNeighbor, valueAtNeighbor);

	  if (!aKnnGraph)
	  {
		  indices1.push_back(outputIndexOfCenterPixel);
		  indices2.push_back(outputIndexOfNeighbor);
	  }
    }
  }

  if (aKnnGraph)
	  GenerateAKnnGraph<double,int>(measurements, tangentLinesPoints1, tangentLinesPoints2, indices1, indices2, NumDimensions, aspectRatio, knnValue, mutualLink, distConstraint, voxelSize);

  BOOST_LOG_TRIVIAL(info) << "measurements.size = " << measurements.size();
  BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints1.size = " << tangentLinesPoints1.size();
  BOOST_LOG_TRIVIAL(info) << "tangentLinesPoints2.size = " << tangentLinesPoints2.size();
  BOOST_LOG_TRIVIAL(info) << "radiuses.size = " << radiuses.size();
  BOOST_LOG_TRIVIAL(info) << "objectnessMeasure.size = " << objectnessMeasure.size();

  BOOST_LOG_TRIVIAL(info) << "indices1.size = " << indices1.size();
  BOOST_LOG_TRIVIAL(info) << "indices2.size = " << indices2.size();

  const std::string measurementsDataSetName = "measurements";
  const std::string tangentLinesPoints1DataSetName = "tangentLinesPoints1";
  const std::string tangentLinesPoints2DataSetName = "tangentLinesPoints2";
  const std::string radiusesDataSetName = "radiuses";
  const std::string objectnessMeasureDataSetName = "objectnessMeasure";

  const std::string indices1DataSetName = "indices1";
  const std::string indices2DataSetName = "indices2";

  FileWriter outputFileWriter(outputFileName);

  outputFileWriter.Write(measurementsDataSetName, measurements);
  outputFileWriter.Write(tangentLinesPoints1DataSetName, tangentLinesPoints1);
  outputFileWriter.Write(tangentLinesPoints2DataSetName, tangentLinesPoints2);
  outputFileWriter.Write(radiusesDataSetName, radiuses);
  outputFileWriter.Write(objectnessMeasureDataSetName, objectnessMeasure);

  outputFileWriter.Write(indices1DataSetName, indices1);
  outputFileWriter.Write(indices2DataSetName, indices2);
}

int main(int argc, char *argv[])
{
  namespace po = boost::program_options;

  std::string inputFileName;
  std::string outputFileName;

  double thresholdBelow = 0.05;
  bool aKnnGraph = false;
  int knnValue = 6;
  double aspectRatio = 0.5;
  bool mutualLink = true;
  double distConstraint = 1.0;
  double voxelSize;

  po::options_description desc;


  desc.add_options()
    ("help", "print usage message")
    ("thresholdBelow", po::value(&thresholdBelow), "the values below the threshold will be ignored")
    ("inputFileName", po::value(&inputFileName)->required(), "the name of the input file")
    ("outputFileName", po::value(&outputFileName)->required(), "the name of the output file")
	("aKnnGraph", po::value(&aKnnGraph)->required(), "the flag indicating the construction of anisotropic KNN graph")
    ("knnValue", po::value(&knnValue), "the number of k nearest neighbor")
	("aspectRatio", po::value(&aspectRatio), "the aspect ratio")
	("mutualLink", po::value(&mutualLink), "mutual connectivity between pair of nodes")
	("distConstraint", po::value(&distConstraint), "anisotropic distance constraint: short axis range in terms of voxelSize")
	("voxelSize", po::value(&voxelSize)->required(), "voxelSize");

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
    DoGenerateNeighborhoodGraph(inputFileName, outputFileName, thresholdBelow, aKnnGraph, aspectRatio, knnValue, mutualLink, distConstraint, voxelSize);
    return EXIT_SUCCESS;
  }
  catch (itk::ExceptionObject& e)
  {
    e.Print(std::cerr);
    return EXIT_FAILURE;
  }
}
