#!/bin/bash

num=$1
dirname="./image$num"

/home/z889zhan/VascularTreeEstimation/VTE-release/Bin/ObjectnessMeasureImageFilter \
--alpha 0.5 \
--beta 0.5 \
--gamma 30 \
--numberOfSigmaSteps 100 \
--scaleObjectnessMeasure true \
--sigmaMinimum 0.023 \
--sigmaMaximum 0.1152 \
--inputFileName "$dirname/GaussianNoiseVolume.mhd" \
--outputFileName "$dirname/ObjectnessMeasureVolume.mhd"

#cat "./thresholdBelow.txt" | xargs -n 1 -I tb -P 4 bash ExprInner.sh $num tb
