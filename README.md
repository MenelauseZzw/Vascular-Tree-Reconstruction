
# Vascular-Tree-Reconstruction (continue to be updated)

## How to use 

 1. Do ObjectenessMasureImageFilter
 
    → ObjectnessMeasureVolume --- data
 
 2. DoNonmaximumSuppressionVolume
 
    → NonMaximumSuppressionVolume.h5 --- graph
 
    Important `--thresholdValue` --- this is threshold for plots
 
 3. DoLevenbergMarquardtMinimizer (use GPU)
 
    → NonMaximumSuppressionCurvVolume.h5
 
    Important `--lambda` --- curvature coefficient
	
	`--beta` --- divergence coefficient
    
    `--voxelPhysicalSize` --- defines allowed error in ||l_p - p||_+
 
 4. DoGenerateMinimumSpanningTree
 
    → NonMaximumSuppressionCurvVolumeEMST.h5 --- tree
 
    Important `--directedLabel`  --- a flag used to decide whether to use minimum arborescence or MST
	
	Code for the minimum arborescence: https://github.com/atofigh/edmonds-alg.git