#ifndef FileWriter_hpp
#define FileWriter_hpp

#ifndef _WIN32
#define FileWriterClassDeclSpec
#endif

#ifndef FileWriterClassDeclSpec
#define FileWriterClassDeclSpec __declspec(dllimport)
#endif

#include <array>
#include <string>
#include <vector>

class FileWriterClassDeclSpec FileWriter
{
public:
  FileWriter(const std::string& fileName);
  ~FileWriter();

  void Write(const std::string& name, double inValue);
  void Write(const std::string& name, int inValue);

  void Write(const std::string& name, const std::vector<double>& inValue);
  void Write(const std::string& name, const std::vector<int>& inValue);

  void Write(const std::string& name, const std::vector<std::array<double, 2U>>& inValue);
  void Write(const std::string& name, const std::vector<std::array<int, 2U>>& inValue);

  void Write(const std::string& name, const std::vector<std::array<double, 3U>>& inValue);
  void Write(const std::string& name, const std::vector<std::array<int, 3U>>& inValue);

  void Write(const std::string& name, const std::vector<std::array<double, 4U>>& inValue);
  void Write(const std::string& name, const std::vector<std::array<int, 4U>>& inValue);

  FileWriter(const FileWriter&) = delete;
  FileWriter& operator=(const FileWriter&) = delete;

private:
  class FileWriterImpl;
  const FileWriterImpl* const pImpl;
};

#endif//FileWriter_hpp
