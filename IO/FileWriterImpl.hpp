#ifndef FileWriterImpl_hpp
#define FileWriterImpl_hpp

#include "FileWriter.hpp"
#include "H5Helper.hpp"

class FileWriter::FileWriterImpl
{
public:
  FileWriterImpl(const std::string& fileName)
    : file(fileName, H5F_ACC_TRUNC)
  {
  }

  template<typename ValueType>
  void Write(const std::string& name, ValueType inValue) const
  {
    const DataSpace space(H5S_SCALAR);
    const PredType type = H5Helper<ValueType>::GetDataType();

    const DataSet ds = file.createDataSet(name.c_str(), type, space);
    ds.write(&inValue, type, space);
  }

  template<typename ValueType>
  void Write(const std::string& name, const std::vector<ValueType>& inValue) const
  {
    const int rank = 1;
    const hsize_t dims = inValue.size();
    const DataSpace space(rank, &dims);
    const PredType type = H5Helper<ValueType>::GetDataType();

    const DataSet ds = file.createDataSet(name.c_str(), type, space);
    ds.write(&inValue[0], type, space);
  }

  template<typename ValueType, size_t NumDimensions>
  void Write(const std::string& name, const std::vector<std::array<ValueType, NumDimensions>>& inValue) const
  {
    const int rank = 2;
    const hsize_t dims[rank] = { inValue.size(), NumDimensions };
    const DataSpace space(rank, dims);
    const PredType type = H5Helper<ValueType>::GetDataType();

    const DataSet ds = file.createDataSet(name.c_str(), type, space);
    ds.write(&inValue[0], type, space);
  }

  FileWriterImpl(const FileWriterImpl&) = delete;
  FileWriterImpl& operator=(const FileWriterImpl&) = delete;

private:
  const H5File file;
};

#endif//FileWriterImpl_hpp