@echo off

set thresholdValue=0.001
set voxelPhysicalSize=0.046

C:\echesakov\VascularTreeEstimation\Bin\NonMaximumSuppressionFilter.exe ^
--inputFileName ".\image001\ObjectnessMeasureVolume.mhd" ^
--outputFileName ".\image001\NonMaximumSuppressionVolume.mhd" ^
--thresholdValue %thresholdValue%

C:\echesakov\VascularTreeEstimation\Bin\GenerateNeighborhoodGraph.exe ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.mhd" ^
--outputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--thresholdValue %thresholdValue%

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\NonMaximumSuppressionVolume.vtp" ^
--pointsArrName "measurements"

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateTangentsPolyDataFile" ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\NonMaximumSuppressionVolumeTangents.vtp" ^
--pointsArrName "measurements" ^
--voxelPhysicalSize %voxelPhysicalSize%

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateRadiusesPolyDataFile" ^
--inputFileName ".\image001\NonMaximumSuppressionVolume.h5" ^
--outputFileName ".\image001\NonMaximumSuppressionVolumeRadiuses.vtp" ^
--pointsArrName "measurements"
