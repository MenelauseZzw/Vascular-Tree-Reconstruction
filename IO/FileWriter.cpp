#include "FileWriter.hpp"
#include "FileWriterImpl.hpp"

FileWriter::FileWriter(const std::string& fileName)
  : pImpl{ new FileWriterImpl(fileName) }
{
}

FileWriter::~FileWriter()
{
  delete pImpl;
}

void FileWriter::Write(const std::string& name, double inValue)
{
  return pImpl->Write(name, inValue);
}

void FileWriter::Write(const std::string& name, int inValue)
{
  return pImpl->Write(name, inValue);
}

#define Comma ,
#define FileWriterWriteFunctionImpl(ValueType)\
void FileWriter::Write(const std::string& name, const ValueType& inValue)\
{\
  return pImpl->Write(name, inValue);\
}

FileWriterWriteFunctionImpl(std::vector<double>)
FileWriterWriteFunctionImpl(std::vector<int>)

FileWriterWriteFunctionImpl(std::vector<std::array<double Comma 2U>>)
FileWriterWriteFunctionImpl(std::vector<std::array<int Comma 2U>>)

FileWriterWriteFunctionImpl(std::vector<std::array<double Comma 3U>>)
FileWriterWriteFunctionImpl(std::vector<std::array<int Comma 3U>>)

FileWriterWriteFunctionImpl(std::vector<std::array<double Comma 4U>>)
FileWriterWriteFunctionImpl(std::vector<std::array<int Comma 4U>>)