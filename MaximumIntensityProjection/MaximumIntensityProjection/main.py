import argparse
from MaximumIntensityProjection import ApplicationBuilder
import os.path
import re
 
def DoMaximumIntensityProjection(args):
    sourceDirName = args.sourceDirName

    pattern = re.compile("^(ScalarVolume\((?P<ScalarVolumeFileName>.+)|VectorVolumeComp\((?P<VectorVolumeFileName>.+),(?P<ComponentIndex>\d+)),(?P<Row>\d+),(?P<Column>\d+)\)$")
    
    builder = ApplicationBuilder()

    for volume in args.volumes:
        m = pattern.match(volume)

        if m is None:
            raise RuntimeError('Command line args')

        row = int(m.group('Row'))
        column = int(m.group('Column'))

        scalarVolumeFileName = m.group('ScalarVolumeFileName')
        if not scalarVolumeFileName is None:
            imageFileName = os.path.join(sourceDirName, scalarVolumeFileName)
            builder.AddScalarVolume(imageFileName, row, column)
        else:
            vectorVolumeFileName = m.group('VectorVolumeFileName')
            if not vectorVolumeFileName is None:
                imageFileName = os.path.join(sourceDirName, vectorVolumeFileName)
                componentIndex = int(m.group('ComponentIndex'))
                builder.AddVectorVolumeComp(imageFileName, componentIndex, row, column)

    pattern = re.compile("^PolyDataFile\((?P<PolyDataFileName>.+),(?P<KeySym>\w),(?P<Red>\d{1,3}),(?P<Green>\d{1,3}),(?P<Blue>\d{1,3})\)$")

    for polyDataFile in args.polyDataFiles:
        m = pattern.match(polyDataFile)

        polyDataFileName = m.group('PolyDataFileName')
        if not polyDataFileName is None:
            keySym = m.group('KeySym')
            red = int(m.group('Red'))
            green = int(m.group('Green'))
            blue = int(m.group('Blue'))

            polyDataFileName = os.path.join(sourceDirName, polyDataFileName)
            builder.AddPolyDataFile(polyDataFileName, keySym, red, green, blue)

    app = builder.BuildApp()
    app.Start()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.set_defaults(func=DoMaximumIntensityProjection)
    argparser.add_argument('--sourceDirName')
    argparser.add_argument('--volumes', nargs='*', default=['ScalarVolume(original_image.mhd,0,0)'])
    argparser.add_argument('--polyDataFiles', nargs='*', default=['PolyDataFile(tree_structure.vtp,1,85,255,127)'])
    args = argparser.parse_args()
    args.func(args)

