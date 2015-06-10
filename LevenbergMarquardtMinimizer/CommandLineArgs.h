#ifndef CommandLineArgs_h
#define CommandLineArgs_h

#include <string>

class CommandLineArgs
{
public:
  const std::string& SourceFileName() const;
  const std::string& ResultFileName() const;
  const std::string& MeasurementsDataSetName() const;
  const std::string& TangentsLinesPoints1DataSetName() const;
  const std::string& TangentsLinesPoints2DataSetName() const;
  const std::string& RadiusesDataSetName() const;
  const std::string& PositionsDataSetName() const;
  const std::string& SourcesDataSetName() const;
  const std::string& TargetsDataSetName() const;
  int NearestNeighbors() const;
  int MaxIterations() const;
  void BriefReport() const;
  static const CommandLineArgs& Instance();
};

#endif//CommandLineArgs_h