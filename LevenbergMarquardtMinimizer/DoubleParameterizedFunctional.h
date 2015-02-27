#ifndef DoubleParameterizedFunctional_h
#define DoubleParameterizedFunctional_h

class DoubleParameterizedFunctional
{
public:
  DoubleParameterizedFunctional();
  double Evaluate(DoubleVector parameters, DoubleVector x);
  void GradientWithRespectToParams(DoubleVector parameters, DoubleVector x, ref DoubleVector grad);
};

#endif//DoubleParameterizedFunctional_h