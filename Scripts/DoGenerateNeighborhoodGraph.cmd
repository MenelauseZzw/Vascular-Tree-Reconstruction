@echo off

set thresholdValue=0.001

C:\echesakov\VascularTreeEstimation\Bin\GenerateNeighborhoodGraph.exe ^
--inputFileName ".\image001\ObjectnessMeasureVolume.mhd" ^
--outputFileName ".\image001\ObjectnessMeasureVolume.h5" ^
--thresholdValue %thresholdValue%

C:\echesakov\VascularTreeEstimation\Bin\GenerateNeighborhoodGraph.exe ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.mhd" ^
--outputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--thresholdValue %thresholdValue%
