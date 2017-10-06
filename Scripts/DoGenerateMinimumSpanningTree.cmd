@echo off

set lambdaValue=1.95

C:\echesakov\VascularTreeEstimation\Bin\GenerateMinimumSpanningTree.exe ^
--knn -1 ^
--optionNum 1 ^
--inputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolume.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.h5"

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe "C:\echesakov\Source\VascularTreeEstimation\Utitilies\main.py" "DoCreateGraphPolyDataFile" ^
--pointsArrName "positions" ^
--inputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.h5" ^
--outputFileName ".\image001\%lambdaValue%\NonMaximumSuppressionCurvVolumeEMST.vtp"
