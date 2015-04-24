#ifndef CommandLineArgs_h
#define CommandLineArgs_h

#include <string>

class CommandLineArgs
{
public:
  static const std::string& SourceFileName();
  static const std::string& ResultFileName();
  static int NearestNeighbors();
  static void Notify();
};

#endif//CommandLineArgs_h