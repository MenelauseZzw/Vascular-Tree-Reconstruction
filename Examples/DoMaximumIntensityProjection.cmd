@echo off

C:\Users\echesako\AppData\Local\Continuum\Anaconda2\python.exe C:\echesakov\Source\VascularTreeEstimation\MaximumIntensityProjection\MaximumIntensityProjection\main.py ^
--sourceDirName ".\image001" ^
--volumes "ScalarVolume(original_image.mhd,0,0)" "VectorVolumeComp(ObjectnessMeasureVolume.mhd,0,0,1)" "VectorVolumeComp(NonMaximumSuppressionVolume.mhd,0,1,0)" ^
--polyDataFiles "PolyDataFile(tree_structure.vtp,1,85,255,127)" "PolyDataFile(ObjectnessMeasureVolumeTangents.vtp,2,255,255,127)" "PolyDataFile(NonMaximumSuppressionVolumeTangents.vtp,3,255,170,255)" "PolyDataFile(1.95/NonMaximumSuppressionCurvVolumeTangents.vtp,4,85,170,255)" "PolyDataFile(1.95/NonMaximumSuppressionCurvVolumeRadiuses.vtp,5,255,85,127)"