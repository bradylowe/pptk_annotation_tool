
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Input filename')
parser.add_argument('--point_size', type=float, default=0.01, help='Point size')
parser.add_argument('--max_points', type=int, default=10000000, help='Number of points to load from file')
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--no-render', dest='render', action='store_false')
parser.set_defaults(render=True)

opt = parser.parse_args()
print(opt)

if __name__ == "__main__":
    """
    When calling this code from command line, a PointCloud object is created with a given input file,
    and then the control of the command line is passed over to the user for an interactive session.
    
    The program can be called like this:
        python3 Main.py --file ~/RackSlice.las --point_size 0.05
    or:
        python3 Main.py -h
    
    In this session, the user can use the point cloud and its functions via pc.
    For example, the user could say:
       mask = pc.select(classes=[1,2,3], blue=range(200, 256))
       pc.render(mask)
       pc.write('~/foo.las', showing=True)
       
    The first line will select all the points which have class 1,2 or 3 (walls, floor, and ceiling), and a
    fairly large blue component (blue ranges from 0 to 255).
    The second line then renders only those points selected in the first line.
    The third line writes these selected points out to a new file called foo.las.
    
    Alternatively, the user could have selected some points using  (ctrl + left-mouse-button) drag and drop, and
    then executed:
        pc.write('~/foo.las', highlighted=True)
        
    This will write a new point cloud file out with only the highlighted points.
    
    Also note that clicking the ']' close square bracket button on the keyboard while viewing the points will
    cycle through the different color schemes of the point cloud including rgb, classes, user_data, and intensity.
    """
    import code
    from PointCloud import PointCloud
    import numpy as np
    import pandas as pd
    import knn as knn
    from Voxelize import VoxelGrid
    import pptk

    pc = PointCloud(opt.file, opt.point_size, opt.max_points, opt.render)
    code.interact(local=locals())

