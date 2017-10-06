@echo off

set lambdaValue=1.95
set voxelPhysicalSize=0.046

mkdir ".\image001\%lambdaValue%"

C:\echesakov\VascularTreeEstimation\Bin\LevenbergMarquardtMinimizer.exe ^
--lambda %lambdaValue% ^
--voxelPhysicalSize %voxelPhysicalSize% ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolume.h5"
