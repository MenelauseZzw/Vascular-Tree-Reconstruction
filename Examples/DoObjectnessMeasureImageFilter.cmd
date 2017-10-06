@echo off

set alpha=0.5
set beta=0.5
set gamma=30.0
set numberOfSigmaSteps=100
set scaleObjectnessMeasure=true
set sigmaMinimum=0.023
set sigmaMaximum=0.1152
set thresholdValue=0.001
set voxelPhysicalSize=0.046

C:\echesakov\VascularTreeEstimation\Bin\ObjectnessMeasureImageFilter.exe ^
--alpha %alpha% ^
--beta %beta% ^
--gamma %gamma% ^
--numberOfSigmaSteps %numberOfSigmaSteps% ^
--scaleObjectnessMeasure %scaleObjectnessMeasure% ^
--sigmaMinimum %sigmaMinimum% ^
--sigmaMaximum %sigmaMaximum% ^
--inputFileName ".\image001\original_image.mhd" ^
--outputFileName ".\image001\ObjectnessMeasureVolume.mhd"

C:\echesakov\VascularTreeEstimation\Bin\GenerateNeighborhoodGraph.exe ^
--inputFileName ".\image001\ObjectnessMeasureVolume.mhd" ^
--outputFileName ".\image001\ObjectnessMeasureVolume.h5" ^
--thresholdValue %thresholdValue%

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--inputFileName ".\image001\ObjectnessMeasureVolume.h5" ^
--outputFileName ".\image001\ObjectnessMeasureVolume.vtp" ^
--pointsArrName "measurements"

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateTangentsPolyDataFile" ^
--inputFileName ".\image001\ObjectnessMeasureVolume.h5" ^
--outputFileName ".\image001\ObjectnessMeasureVolumeTangents.vtp" ^
--pointsArrName "measurements" ^
--voxelPhysicalSize %voxelPhysicalSize%