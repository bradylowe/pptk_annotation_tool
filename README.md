# PointBluePython

This repo implements algorithms for converting Point Cloud scans of warehouse rooms into 
blueprints of those rooms. The goal is to locate important objects such as walls, floor, ceiling, 
support poles, door frames, racks, pallets, etc for the purpose of virtualizing and visualizing the
warehouse.

## Requirements

0) python - Versions 3.5+ are currently supported (it may be difficult to install pptk with python 3.8)
1) numpy - Required for performing mathematical functions on point cloud data
2) laspy - Required for reading and writing point cloud data to and from LAS files
3) open3d - Required for reading and writing data to and from pcd and ply files
4) pptk - Required for visualizing and manipulating point cloud data by hand
5) Annoy - Used to calculate nearest neighbor information in point cloud (not necessary per se)
6) scipy - Used to calculate nearest neighbor information in the point cloud
7) code - Used to enter interactive sessions from python scripts
8) hdbscan - Used as a clustering algorithm (point segmentation)
9) pandas - Used along with numpy to efficiently manipulate and store point cloud data structures
10) sklearn - Used for KMeans algorithm for another point clustering method

If you want to read/write directly from/to zipped laz files, then you will need to have installed the laszip
command line tool.

## Quick Start Guide (Instructions)

Clone the repository and navigate to the new directory.

To install above requirements, run `pip install -r requirements.txt`

Note that if a pptk wheel file is not available for your version of python, it may be possible to download another pptk 
wheel file from an older version of python, rename the wheel to reflect your current version of python, and install it 
manually using pip install wheel_file_name.whl. Wheel files can be found
[here](https://pypi.org/project/pptk/#files). A wheel file for mac users using python 3.7 is included in this repo.

Then finally, to render a PointCloud file run:

```python
python3 Main.py --file /path/to/las/file.las --point_size 0.03 --max_points 100000
```

This will load the LAS file into a PointCloud object named "pc" along with a pptk viewer window 
and hand over control of an interactive session to you in the console. From here, you can 
use the many commands of the project. 

## PPTK Point Cloud Viewer Window

First of all, try manipulating the point cloud in the viewer. You can:

 - Rotate the view using the left-mouse-button (LMB) and dragging/dropping
 - Zoom in and out using the mouse-scroll-wheel
 - Pan the view by using shift + LMB and drag/drop
 - Highlight points using ctrl + LMB and drag/drop (or click a single point to highlight it)
 - You can un-highlight points by using shift + ctrl + LMB and drag/drop (or single click a point)
 - Use the numerical keys 1, 3, 7 to look along the x, y, and z directions respectively
 - Use the numerical 5 key to toggle between orthographic and perspective views
 - Use the square bracket keys "[  ]" to cycle through the different coloring schemes of the points
    * The viewer opens up with the points colored by RGB data
    * Clicking the "]" key will show the points colored by classification data (if it exists)
    * Clicking the "]" key again will show the points colored by user_data (if it exists)
    * Clicking the "]" one last time will show the points colored by intensity data (if it exists)

## Custom designed (command line) functionality

Once we have some points highlighted, we can call various functions from the command line to do 
further manipulation. For instance, after highlighting some points, type into the python prompt:

```python
pc.render(highlighted=True)
```

This will clear the viewer of all points and load only those points that were highlighted into 
the viewer so that you can focus on those points. You can then highlight a subset of these newly
displayed points and issue the command again for a similar effect. 

If instead you want to save that point selection for later, you can type

```python
mask = pc.select()
```

This will store the point selection in a boolean mask called "mask". There are various things
you could do with mask. For instance, you can type the following commands:

```python
mask = pc.select()  # Store the highlighted selection
points = pc.points.loc[mask][['x', 'y', 'z']]  # Put the selected points into their own array
centroid = Features.centroid(points)  # Calculate the centroid of the selected points
bbox = Features.bounding_box(points)  # Calculate the bounding box of the selected points
normal, curvature = Features.normal(points, curvature=True)  # Estimate the normal direction and curvature
pc.render(mask)  # Render the selection
pc.render()  # Render all the points in the point cloud
pc.highlight(mask)  # Highlight the original selection
count = np.sum(mask)  # Count how many points in the selection
count_rack = np.sum(pc.points.loc[mask, 'class'] == 4)  # Count how many of the selected points are rack 

pc.classify(4, mask=mask)  # If the points are already classified, this line does nothing
pc.classify(4, True, mask)  # This will overwrite the classification of the given points as 4 (rack)
pc.write(pc.filename.replace('.las', '_subset.las'), mask=mask)  # Write the selected points to a new LAS file
```

## Point Selection

Using orthographic view (toggle with numeric 5 key) while looking down a particular axis allows you to quickly and efficiently
select points that run along the axis such as rack, walls, floor, ceiling, etc. Switching to 
perspective view allows for a more natural view to examine and refine the selection. This allows
the user to quickly and easily classify objects in the scene. 

Besides hand selection, there is another way to select certain points:  the select() method. This 
method has many possible selection criteria, and when the user types in more than one, only the
points that satisfy ALL of the criteria will be returned. 

Here is a list of the input parameters along with a summary of their use:
 
 - indices - You may pass in a list of point indices to select
 - highlighted - If True, then the points highlighted in the viewer will be selected (default True)
 - showing - If True, then the currently rendered points will be selected (default False)
 - classes - You may pass in a list of classes to be chosen (can use range())
    * Example:  `pc.select(classes=range(3, 6))` will select all points with class 3, 4, or 5
    * Example:  `pc.select(classes=[4, 7, 9])`
    * Example:  `pc.select(classes=[5])`
    * classes always lie in the range of 0 to 31
 - data - You may pass in a list of possible user_data values to be selected (can use range())
    * Example:  `pc.select(data=range(150, 256))` 
    * user_data usually ranges from 0 to 255 (can range from 0 to 65,535)
 - intensity - Same as user_data
 - red - Again, same as user_data. We can select points whose red value lies in a certain range
    * red, green, and blue usually lie in the range 0 to 255, but could go to 65,535
 - green - See "red"
 - blue - See "red"
 - compliment - If True, then the result of the selection will be reversed before returning

## Writing LAS Files

The `pc.write()` function has a few options for writing points to file:

 - filename - New output LAS filename. If None, then {origFilename}_out.las will be used
 - mask - Optionally send in a boolean mask to indicate which points to write
 - indices - Optionally send in a list of indices to indicate which to write
 - highlighted - If True, then write out all the highlighted points
 - showing - If True, then write out all the currently rendered points
 - overwrite - If True (and no filename specified), then overwrite {origFilename}.las
 - points - Optionally send in a pandas DataFrame containing columns 'x', 'y', 'z', and optionally 'class', 'user_data', and 'intensity' to write to file
 
## Voxelization

The Voxelize.py file has code for sorting the 3D point cloud data spatially. Simply create a 
VoxelGrid object by calling 

```python
vg = Voxelize.VoxelGrid(pc.points, meshSize=0.02)
```

and then use the "vg" to make nearest neighbor queries such as 

```python
point = pc.points[0]
voxels = vg.neighbors(point, radius=0.1) # Get voxels within 10 cm of the given point
voxels = vg.neighbors(point, overlap=3) # Get all voxels within 3 voxels of the center voxel.
neighbors = vg.contents(voxels, merge=True) # Get the points contained in the list of voxels returned from the neighbors function
```
