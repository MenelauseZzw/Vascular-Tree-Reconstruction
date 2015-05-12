#include "TestKnnSearch.h"
#include <flann/flann.hpp>

void testKnnSearch(float* pP, const float *pSigma, const float* pS, const float* pT, int* pIndPi, int* pIndPj, int numPoints, int nn, int& numPairs)
{
  using namespace flann;

  nn++;

  Matrix<float> dataset(pP, numPoints, 3);
  Matrix<float> query(pP, numPoints, 3);

  // construct an randomized kd-tree index using 4 kd-trees
  Index<L2<float> > index(dataset, flann::KDTreeIndexParams(1));
  index.buildIndex();

  Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
  Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

  // do a knn search, using 128 checks
  index.knnSearch(query, indices, dists, nn, flann::SearchParams(-1));

  //flann::SearchParams params;
  //params.checks = -1;
  //params.max_neighbors = nn;

  //const float radius = 3;
  //index.radiusSearch(query, indices, dists, radius, params);

  numPairs = 0;
  for (int i = 0; i < query.rows; ++i)
  {
    for (int k = 0; k < nn; ++k)
    {
      if (indices[i][k] == -1) break;

      int j = indices[i][k];
      /* if ((i < j) && (dists[i][k] < pSigma[i] + pSigma[j]))
       {
       indPi.push_back(i);
       indPj.push_back(j);
       }*/
      if (i != j)
      {
        float piMinusPj[3];
        float siMinusTi[3];
        float sjMinusTj[3];

        for (int n = 0; n < 3; ++n)
        {
          piMinusPj[n] = dataset[i][n] - dataset[j][n];
          siMinusTi[n] = pS[3 * i + n] - pT[3 * i + n];
          sjMinusTj[n] = pS[3 * j + n] - pT[3 * j + n];
        }

        float siMinusTiCrossSjMinusTj[3];

        siMinusTiCrossSjMinusTj[0] = siMinusTi[1] * sjMinusTj[2] - siMinusTi[2] * sjMinusTj[1]; //s[1] = u[2]v[3] - u[3]v[2]
        siMinusTiCrossSjMinusTj[1] = siMinusTi[2] * sjMinusTj[0] - siMinusTi[0] * sjMinusTj[2]; //s[2] = u[3]v[1] - u[1]v[3]
        siMinusTiCrossSjMinusTj[2] = siMinusTi[0] * sjMinusTj[1] - siMinusTi[1] * sjMinusTj[0]; //s[3] = u[1]v[2] - u[2]v[1]

        float siMinusTiCrossSjMinusTjSq = 0;
        for (int n = 0; n < 3; ++n)
        {
          siMinusTiCrossSjMinusTjSq += siMinusTiCrossSjMinusTj[n] * siMinusTiCrossSjMinusTj[n];
        }

        float invSiMinusTiCrossSjMinusTj = 1 / sqrt(siMinusTiCrossSjMinusTjSq);

        float distanceBetweenLines = 0;
        for (int n = 0; n < 3; ++n)
        {
          distanceBetweenLines += piMinusPj[n] * siMinusTiCrossSjMinusTj[n];
        }

        distanceBetweenLines = abs(distanceBetweenLines);
        distanceBetweenLines *= invSiMinusTiCrossSjMinusTj;

        if (true || /*2 **/ distanceBetweenLines < pSigma[i] + pSigma[j])
        {
          pIndPi[numPairs] = i;
          pIndPj[numPairs] = j;

          ++numPairs;
        }
      }
    }
  }

  //delete[] dataset.ptr();
  //delete[] query.ptr();
  delete[] indices.ptr();
  delete[] dists.ptr();
}