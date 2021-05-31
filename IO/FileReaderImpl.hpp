#ifndef FileReaderImpl_hpp
#define FileReaderImpl_hpp

#include "FileReader.hpp"
#include "H5Helper.hpp"

class FileReader::FileReaderImpl
{
public:
  FileReaderImpl(const std::string& fileName)
    : file(fileName, H5F_ACC_RDONLY)
  {
  }

  template<typename ReturnValueType>
  ReturnValueType ReadValue(const std::string& name) const
  {
    const DataSet ds = file.openDataSet(name);

    ReturnValueType outValue;
    ds.read(&outValue, H5Helper<ReturnValueType>::GetDataType(), DataSpace(H5S_SCALAR));
    return outValue;
  }

  template<typename ValueType>
  void Read(const std::string& name, std::vector<ValueType>& outValue) const
  {
    const DataSet ds = file.openDataSet(name);
    const int rank = 1;
    const DataSpace space = ds.getSpace();

    if (space.getSimpleExtentNdims() != rank)
      throw std::invalid_argument("outValue");

    outValue.resize(space.getSimpleExtentNpoints());
    ds.read(&outValue[0], H5Helper<ValueType>::GetDataType(), space);
  }

  template<typename ValueType, size_t NumDimensions>
  void Read(const std::string& name, std::vector<std::array<ValueType, NumDimensions>>& outValue) const
  {
    const DataSet ds = file.openDataSet(name);
    const int rank = 2;
    const DataSpace space = ds.getSpace();

    if (space.getSimpleExtentNdims() != rank)
      throw std::invalid_argument("outValue");

    hsize_t dims[rank];
    space.getSimpleExtentDims(dims);

    if (dims[1] != NumDimensions)
      throw std::invalid_argument("outValue");

    outValue.resize(dims[0]);
    ds.read(&outValue[0], H5Helper<ValueType>::GetDataType(), space);
  }

  FileReaderImpl(const FileReaderImpl&) = delete;
  FileReaderImpl& operator=(const FileReaderImpl&) = delete;

private:
  const H5File file;
};

#endif//FileReaderImpl_hpp