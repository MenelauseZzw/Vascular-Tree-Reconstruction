
# Vascular-Tree-Reconstruction (to be updated)

## How to use 

 1. Do ObjectenessMasureImageFilter (Frangi filtering)
 
    → ObjectnessMeasureVolume --- volume data
 
 2. Do NonmaximumSuppressionVolume
 
    → NonMaximumSuppressionVolume --- volume data
 
    Important `--thresholdValue` --- this is threshold for plots
	
 3. DO GenerateNeighborhoodGraph
 
    → NonMaximumSuppressionVolume.h5 --- graph
 
 4. Do LevenbergMarquardtMinimizer (use GPU)
 
    → NonMaximumSuppressionCurvDivVolume.h5
 
    Important `--lambda` --- curvature coefficient
	
	`--beta` --- divergence coefficient
    
    `--voxelPhysicalSize` --- defines allowed error in ||l_p - p||_+
 
 5. Do GenerateTreeTopology
 
    → NonMaximumSuppressionCurvDivVolumeTree.h5 --- tree
 
    Important `--directedLabel`  --- a flag used to decide whether to use minimum arborescence or MST
	
	Code for the minimum arborescence: https://github.com/atofigh/edmonds-alg.git