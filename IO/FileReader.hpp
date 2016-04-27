#ifndef FileReader_hpp
#define FileReader_hpp

#ifndef FileReaderClassDeclSpec
#define FileReaderClassDeclSpec __declspec(dllimport)
#endif

#include <array>
#include <string>
#include <vector>

class FileReaderClassDeclSpec FileReader
{
public:
  FileReader(const std::string& fileName);
  ~FileReader();

  double ReadDouble(const std::string& name) const;
  int ReadInt(const std::string& name) const;

  void Read(const std::string& name, std::vector<double>& outValue) const;
  void Read(const std::string& name, std::vector<int>& outValue) const;

  void Read(const std::string& name, std::vector<std::array<double, 2U>>& outValue) const;
  void Read(const std::string& name, std::vector<std::array<int, 2U>>& outValue) const;

  void Read(const std::string& name, std::vector<std::array<double, 3U>>& outValue) const;
  void Read(const std::string& name, std::vector<std::array<int, 3U>>& outValue) const;

  void Read(const std::string& name, std::vector<std::array<double, 4U>>& outValue) const;
  void Read(const std::string& name, std::vector<std::array<int, 4U>>& outValue) const;

  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;

private:
  class FileReaderImpl;
  const FileReaderImpl* const pImpl;
};

#endif//FileReader_hpp