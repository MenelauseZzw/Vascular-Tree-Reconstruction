import argparse
import IO
from LevenbergMarquardtMinimizer import InexactLevenbergMarquardtMinimizer,ProblemBuilder

def DoLevenbergMarquardtMinimizer(args):
    inputFileName     = args.inputFileName
    outputFileName    = args.outputFileName
    lambdaValue       = args.lambdaValue
    voxelPhysicalSize = args.voxelPhysicalSize

    dataset = IO.ReadFile(inputFileName)

    measurements        = dataset['measurements']
    tangentLinesPoints1 = dataset['tangentLinesPoints1']
    tangentLinesPoints2 = dataset['tangentLinesPoints2']
    radiuses            = dataset['radiuses']
    objectnessMeasure   = dataset['objectnessMeasure']
    indices1            = dataset['indices1']
    indices2            = dataset['indices2']

    builder = ProblemBuilder(measurements, tangentLinesPoints1, tangentLinesPoints2, radiuses, indices1, indices2, lambdaValue, voxelPhysicalSize)
    minimizer = InexactLevenbergMarquardtMinimizer(builder)

    tangentLinesPoints1, tangentLinesPoints2, positions = minimizer.minimize()

    dataset['tangentLinesPoints1'] = tangentLinesPoints1
    dataset['tangentLinesPoints2'] = tangentLinesPoints2
    dataset['positions']           = positions

    IO.WriteFile(outputFileName, dataset)

if __name__ == '__main__':
    # create the top-level parser
    argparser = argparse.ArgumentParser()
    
    argparser.set_defaults(func=DoLevenbergMarquardtMinimizer)
    argparser.add_argument('--inputFileName')
    argparser.add_argument('--outputFileName')
    argparser.add_argument('--lambdaValue', type=float)
    argparser.add_argument('--voxelPhysicalSize', type=float)

    # parse the args and call whatever function was selected
    args = argparser.parse_args()
    print('inputFileName = \"{args.inputFileName}\"\n'
        'outputFileName = \"{args.outputFileName}\"\n'
        'lambdaValue={args.lambdaValue}\n'
        'voxelPhysicalSize={args.voxelPhysicalSize}'.format(args=args))
    args.func(args)