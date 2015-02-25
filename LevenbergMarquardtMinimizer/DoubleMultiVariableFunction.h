#ifndef DoubleMultiVariableFunction_h
#define DoubleMultiVariableFunction_h

class DoubleMultiVariableFunction
{
public:
  DoubleMultiVariableFunction();
  void Evaluate(double x[]);
  void Jacobian(double x[], double jac[]);
};

#endif//DoubleMultiVariableFunction_h