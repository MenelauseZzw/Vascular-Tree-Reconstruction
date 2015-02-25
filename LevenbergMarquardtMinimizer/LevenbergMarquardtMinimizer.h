#ifndef LevenbergMarquardtMinimizer_h
#define LevenbergMarquardtMinimizer_h

class DoubleMultiVariableFunction;

class LevenbergMarquardtMinimizer
{
public:
  LevenbergMarquardtMinimizer(int maxIterations, double gradientTolerance, double solutionDeltaTolerance);
  double Minimize(DoubleMultiVariableFunction& f, double x0[]);
  double FinalResidual() const;
  double GradientTolerance() const;
  double InitialResidual() const;
  int Iterations() const;
  int MaxIterations() const;
  bool MaxIterationsMet() const;
  double SolutionDeltaTolerance() const;
  double Tau() const;
};

#endif//LevenbergMarquardtMinimizer_h