#include "FileReader.hpp"
#include "FileReaderImpl.hpp"

FileReader::FileReader(const std::string& fileName)
  : pImpl{ new FileReaderImpl(fileName) }
{
}

FileReader::~FileReader()
{
  delete pImpl;
}

double FileReader::ReadDouble(const std::string& name) const
{
  return pImpl->ReadValue<double>(name);
}

int FileReader::ReadInt(const std::string& name) const
{
  return pImpl->ReadValue<int>(name);
}

#define Comma ,
#define FileReaderReadFunctionImpl(ValueType)\
void FileReader::Read(const std::string& name, ValueType& outValue) const\
{\
  return pImpl->Read(name, outValue);\
}

FileReaderReadFunctionImpl(std::vector<double>)
FileReaderReadFunctionImpl(std::vector<int>)

FileReaderReadFunctionImpl(std::vector<std::array<double Comma 2U>>)
FileReaderReadFunctionImpl(std::vector<std::array<int Comma 2U>>)

FileReaderReadFunctionImpl(std::vector<std::array<double Comma 3U>>)
FileReaderReadFunctionImpl(std::vector<std::array<int Comma 3U>>)

FileReaderReadFunctionImpl(std::vector<std::array<double Comma 4U>>)
FileReaderReadFunctionImpl(std::vector<std::array<int Comma 4U>>)