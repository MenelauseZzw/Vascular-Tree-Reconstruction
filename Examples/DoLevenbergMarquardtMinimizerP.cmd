@echo off

set lambdaValue=1.95
set voxelPhysicalSize=0.046

mkdir ".\image001\%lambdaValue%"

C:\Users\echesako\AppData\Local\Continuum\Anaconda3\envs\tensorflow\python.exe C:\echesakov\Source\VascularTreeEstimation\LevenbergMarquardtMinimizer\LevenbergMarquardtMinimizer\main.py ^
--lambdaValue %lambdaValue% ^
--voxelPhysicalSize %voxelPhysicalSize% ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolume.h5"
