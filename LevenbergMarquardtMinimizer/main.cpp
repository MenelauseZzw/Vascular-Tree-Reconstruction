#include <H5Cpp.h>

#ifndef H5_NO_NAMESPACE

using H5::H5File;
using H5::DataSpace;
using H5::PredType;

#endif



int main(int argc, char *argv[])
{
  H5File sourceData("C:\\WesternU\\test.h5", H5F_ACC_RDONLY);
  auto set = sourceData.openDataSet("~p.x");
  auto space = set.getSpace();


  hsize_t ndims;
  space.getSimpleExtentDims(&ndims);

}