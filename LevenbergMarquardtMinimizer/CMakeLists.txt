cmake_minimum_required(VERSION 2.8.11)
project(LevenbergMarquardtMinimizer)

add_library(
  LevenbergMarquardtMinimizer
  DoubleMultiVariableFunction.cpp
  DoubleMultiVariableFunction.h
  LevenbergMarquardtMinimizer.cpp
  LevenbergMarquardtMinimizer.h
  )

