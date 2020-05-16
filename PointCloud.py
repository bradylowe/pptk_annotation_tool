from laspy.file import File
from laspy.header import Header
import open3d as o3d
import numpy as np
import pandas as pd
import pptk
import os
from tqdm import tqdm
from collections import defaultdict

from Mask import Mask
import knn as knn
from Voxelize import VoxelGrid


class PointCloud:
    def __init__(self, filename=None, point_size=0.01, max_points=10000000, render=True):
        self.point_size = point_size
        self.max_points = max_points
        self.render_flag = render
        self.viewer = None
        self.las_header = None
        self.points = pd.DataFrame(columns=['x', 'y', 'z', 'class'])
        self.showing = None
        self.index = None
        self.filename = filename
        if filename is None:
            self.render_flag = False
        else:
            self.load(filename)

    def __del__(self):
        if self.viewer:
            self.viewer.close()

    def __len__(self):
        return len(self.points)

    def load(self, filename, max_points=None):
        if max_points is not None:
            self.max_points = max_points
        if filename.endswith('.las') or filename.endswith('.laz'):
            self.__load_las_file(filename)
        elif filename.endswith('.ply'):
            self.__load_ply_file(filename)
        elif filename.endswith('.pcd'):
            self.__load_pcd_file(filename)
        elif filename.endswith('.xyz') or filename.endswith('.pts') or filename.endswith('.txt'):
            self.__load_xyz_file(filename)
        else:
            print('Cannot load %s: file type not supported' % filename)
            return
        self.showing = Mask(len(self.points), True)
        self.render(self.showing)

    def __from_open3d_point_cloud(self, cloud):
        new_df = pd.DataFrame(np.asarray(cloud.points), columns=['x', 'y', 'z'])
        if cloud.has_normals():
            normals = np.asarray(cloud.normals)
            new_df['r'] = (normals[:, 0] * 255.).astype(int)
            new_df['g'] = (normals[:, 1] * 255.).astype(int)
            new_df['b'] = (normals[:, 2] * 255.).astype(int)
        if cloud.has_colors():
            colors = np.asarray(cloud.colors)
            if colors[:, 0].max() > 0:
                new_df['class'] = (colors[:, 0] * 31.).astype(int)
            if colors[:, 1].max() > 0:
                new_df['user_data'] = (colors[:, 1] * 255.).astype(int)
            if colors[:, 2].max() > 0:
                new_df['intensity'] = (colors[:, 2] * 255.).astype(int)
        if self.max_points is not None and self.max_points < len(new_df):
            new_df = new_df.loc[np.random.choice(len(new_df), self.max_points)]
        return new_df

    def __unzip_laz(self, infile, outfile=None):
        import subprocess
        if outfile is None:
            outfile = infile.replace('.laz', '.las')
        args = ['laszip', '-i', infile, '-o', outfile]
        subprocess.run(" ".join(args), shell=True, stdout=subprocess.PIPE)

    def __zip_las(self, infile, outfile=None):
        import subprocess
        if outfile is None:
            outfile = infile.replace('.las', '.laz')
        args = ['laszip', '-i', infile, '-o', outfile]
        subprocess.run(" ".join(args), shell=True, stdout=subprocess.PIPE)

    def __load_las_file(self, filename):
        if filename.endswith('.laz'):
            orig_filename = filename
            filename = 'TEMPORARY.las'
            self.__unzip_laz(orig_filename, filename)
        with File(filename) as f:
            if self.las_header is None:
                self.las_header = f.header.copy()
            if self.max_points is not None and self.max_points < f.header.point_records_count:
                mask = Mask(f.header.point_records_count, False)
                mask[np.random.choice(f.header.point_records_count, self.max_points)] = True
            else:
                mask = Mask(f.header.point_records_count, True)
            new_df = pd.DataFrame(np.array((f.x, f.y, f.z)).T[mask.bools])
            new_df.columns = ['x', 'y', 'z']
            if f.header.data_format_id >= 2:
                rgb = pd.DataFrame(np.array((f.red, f.green, f.blue), dtype='int').T[mask.bools])
                rgb.columns = ['r', 'g', 'b']
                new_df = new_df.join(rgb)
            new_df['class'] = f.classification[mask.bools]
            if np.sum(f.user_data):
                new_df['user_data'] = f.user_data[mask.bools].copy()
            if np.sum(f.intensity):
                new_df['intensity'] = f.intensity[mask.bools].copy()
        self.points = self.points.append(new_df, sort=False)
        if filename == 'TEMPORARY.las':
            os.system('rm TEMPORARY.las')

    def __load_ply_file(self, filename):
        points = o3d.io.read_point_cloud(filename)
        self.points = self.points.append(self.__from_open3d_point_cloud(points), sort=False)

    def __load_xyz_file(self, filename):
        """
        This function allows the user to load point cloud data from an ascii text file format (extension xyz,
        txt, csv, etc.). The text file must have a header as the first line labeling the columns.
        """
        with open(filename) as f:
            # Find out if the delimiter is a comma or a space
            first_line = f.readline()
            split_first_line = first_line.split(',')
            if split_first_line[0] == first_line:
                split_first_line = first_line.split(' ')
                # Find out if the delimiter is something other than a comma or space
                if split_first_line[0] == first_line:
                    print('Unsupported delimiter')
                    return
                delimiter = ' '
            else:
                delimiter = ','
            # Find out if this file has a header as the first line, and if so, grab the names from it
            has_header = first_line.lower().islower()
            if has_header:
                header = 0
                names = split_first_line
            else:
                header = None
                names = ['x', 'y', 'z', 'class', 'r', 'g', 'b']
                names = names[:len(split_first_line)]
            # Find out if the first column of the data is row indices
            if has_header:
                has_indices = not len(split_first_line[0])
            else:
                has_indices = False
            # If the first column does represent row indices, remove that column name
            if has_header and has_indices:
                names = names[1:]

        # Read the csv-like file
        if has_indices:
            new_df = pd.DataFrame.from_csv(filename, sep=delimiter)
        elif has_header:
            new_df = pd.read_csv(filename, delimiter=delimiter, header=header)
        else:
            new_df = pd.read_csv(filename, delimiter=delimiter, header=header, names=names)

        # Make sure we have x and y point values, and make sure we have z values and class values
        if 'x' not in new_df or 'y' not in new_df:
            if 'x' not in new_df or 'y' not in new_df:
                print('Error:  x and/or y missing from dataset. Please make sure there is x and y data in the point cloud',
                  'file, and that the file header indicates which columns store which attribute.')
                #return
        if 'z' not in new_df:
            self.points['z'] = np.zeros(len(self.points))
        print(new_df.columns)
        new_df.columns = ['x', 'y', 'z']
        new_df['class'] = np.zeros(len(new_df), dtype=int)
        self.points = self.points.append(new_df, sort=False)

    def __load_pcd_file(self, filename):
        points = o3d.io.read_point_cloud(filename)
        self.points = self.points.append(self.__from_open3d_point_cloud(points), sort=False)

    def __to_open3d_point_cloud(self, df):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
        if 'r' in df.columns:
            cloud.normals = o3d.utility.Vector3dVector(df[['r', 'g', 'b']].values / 255.)
        colors = np.zeros((len(df), 3))
        colors[:, 0] = df['class'] / 31.
        if 'user_data' in df.columns:
            colors[:, 1] = df['user_data'] / 255.
        if 'intensity' in df.columns:
            colors[:, 2] = df['intensity'] / 255.
        if colors.max() > 0:
            cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def write(self, filename=None, mask=None, indices=None, highlighted=False, showing=False, overwrite=False, points=None):
        """
        This function allows the user to write out a subset of the current points to a file.
        :param filename: Output filename to write to. Default is {current_filename}_out.las.
        :param mask: Mask object used to indicate which points to write to file.
        :param indices: List of integer indices indicating which points to write to file.
        :param highlighted: If True, then write the currently highlighted points to file.
        :param showing: If True, then write all the currently rendered points to file.
        :param overwrite: If True, then overwrite an existing file
        :param points: Pandas DataFrame containing the points and all the data to write. This DataFrame object must
        have x, y, and z attributes and optionally can have r, g, b, class, intensity, and user_data attributes.
        """
        if filename is None:
            filename = self.filename
        if os.path.exists(filename) and not overwrite:
            print(filename, 'already exists. Use option "overwrite=True" to overwrite')
            return
        if mask is None and points is None:
            mask = self.select(indices, highlighted, showing)
        if points is None:
            points = self.points.loc[mask.bools]
        if filename.endswith('.las'):
            self.__write_las_file(filename, points)
        elif filename.endswith('.laz'):
            self.__write_laz_file(filename, points)
        elif filename.endswith('.ply'):
            self.__write_ply_file(filename, points)
        elif filename.endswith('.pcd'):
            self.__write_pcd_file(filename, points)
        elif filename.endswith('.xyz') or filename.endswith('.pts') or filename.endswith('.txt'):
            self.__write_xyz_file(filename, points)
        else:
            print('Unrecognized file type. Please use .las, .ply, .pcd, .xyz, .pts, or .txt.')
            return
        print('Wrote %d points to %s' % (len(points), filename))

    def __write_laz_file(self, filename, points):
        self.__write_las_file('TEMPORARY.las', points)
        self.__zip_las('TEMPORARY.las', filename)
        os.system('rm TEMPORARY.las')

    def __write_las_file(self, filename, points):
        if self.las_header is None:
            self.las_header = Header()
            self.las_header.x_offset, self.las_header.y_offset, self.las_header.z_offset = 0.0, 0.0, 0.0
            self.las_header.x_scale, self.las_header.y_scale, self.las_header.z_scale = 0.0001, 0.0001, 0.0001
        if self.las_header.data_format_id < 2:
            self.las_header.data_format_id = 2
        with File(filename, self.las_header, mode='w') as f:
            f.x, f.y, f.z = points[['x', 'y', 'z']].values.T
            if 'r' in points:
                f.red, f.green, f.blue = points[['r', 'g', 'b']].values.T
            if 'class' in points:
                f.classification = points['class'].values.astype(int)
            if 'user_data' in points:
                f.user_data = points['user_data'].values.astype(int)
            if 'intensity' in points:
                f.intensity = points['intensity'].values.astype(int)

    def __write_xyz_file(self, filename, points):
        points.to_csv(filename)

    def __write_ply_file(self, filename, points):
        cloud = self.__to_open3d_point_cloud(points)
        o3d.io.write_point_cloud(filename, cloud)

    def __write_pcd_file(self, filename, points):
        cloud = self.__to_open3d_point_cloud(points)
        o3d.io.write_point_cloud(filename, cloud)

    def prepare_viewer(self, render_flag=None):
        """
        Check to see if the viewer is ready to receive commands. If it isn't, get it ready and return True.
        If the render flag is False, return False.
        """
        if render_flag is not None:
            self.render_flag = render_flag
        if not self.render_flag:
            return False
        if not self.viewer_is_ready():
            self.render(showing=True)
            return True

    def viewer_is_ready(self):
        """
        Return True if the viewer is ready to receive commands, else return False.
        """
        if not self.render_flag or not self.viewer:
            return False
        try:
            self.viewer.get('lookat')
            return True
        except ConnectionRefusedError:
            return False

    def close_viewer(self):
        if self.viewer_is_ready():
            self.viewer.close()
        self.viewer = None

    def render(self, mask=None, indices=None, highlighted=False, showing=False):
        """
        This function allows the user to render some selection of the points to the viewer.
        By default, this function will render all points when called. If a mask is supplied, then those points
        will be rendered. If the highlighted or showing flags are True, then the appropriate selection will be used.
        :param mask: Mask object indicating which points to render.
        :param highlighted: If True, then render the currently highlighted points.
        :param showing: If True, then re-render all of the currently rendered points.
        """
        if not self.render_flag:
            return

        if mask is None:
            mask = self.select(indices=indices, highlighted=highlighted, showing=showing)

        if not np.sum(mask[:]):
            return

        self.showing.set(mask[:])

        if self.viewer_is_ready():
            self.viewer.clear()
            self.viewer.load(self.points.loc[mask[:], ['x', 'y', 'z']])
        else:
            self.viewer = pptk.viewer(self.points.loc[mask[:]][['x', 'y', 'z']])

        self.viewer.set(point_size=self.point_size, selected=[])
        if 'r' in self.points:
            scale = 255.0
            if 'user_data' in self.points and 'intensity' in self.points:
                self.viewer.attributes(self.points.loc[mask[:], ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask[:], 'class'],
                                       self.points.loc[mask[:], 'user_data'],
                                       self.points.loc[mask[:], 'intensity'])
            elif 'user_data' in self.points:
                self.viewer.attributes(self.points.loc[mask[:], ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask[:], 'class'],
                                       self.points.loc[mask[:], 'user_data'])
            elif 'intensity' in self.points:
                self.viewer.attributes(self.points.loc[mask[:], ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask[:], 'class'],
                                       self.points.loc[mask[:], 'intensity'])
            else:
                self.viewer.attributes(self.points.loc[mask[:], ['r', 'g', 'b']] / scale,
                                       self.points.loc[mask[:], 'class'])
        else:
            self.viewer.attributes(self.points.loc[mask[:], 'class'])

    def get_relative_indices(self, mask, relative=None):
        """
        Return the chosen point indices relative to the currently rendered points (or some other set).
        """
        if relative is None:
            relative = self.showing
        mask.bools = mask.bools[relative.bools]
        return mask.resolve()

    def get_highlighted_mask(self):
        """
        Return a mask indicating which points are currently highlighted in the viewer.
        """
        mask = Mask(len(self.points), False)
        if self.viewer_is_ready():
            mask.setr_subset(self.viewer.get('selected'), self.showing)
        return mask

    def highlight(self, mask=None, indices=None):
        """
        Set the selected points to be highlighted in the viewer (if they are rendered).
        """
        if mask is None and indices is None:
            return
        if indices is not None:
            mask = self.select(indices=indices, highlighted=False)
        if self.viewer_is_ready():
            indices = self.get_relative_indices(mask)
            self.viewer.set(selected=indices)

    def get_perspective(self):
        """
        This function captures the current perspective of the viewer and returns its parameters so that the user
        can return to this perspective later or use it in a rendering sequence.
        :return: Perspective parameters (x, y, z, phi, theta, r).
        """
        if not self.viewer_is_ready():
            return [0, 0, 0, 0, 0, 0]
        x, y, z = self.viewer.get('eye')
        phi = self.viewer.get('phi')
        theta = self.viewer.get('theta')
        r = self.viewer.get('r')
        return [x, y, z, phi, theta, r]

    def set_perspective(self, p):
        """
        This method allows the user to set the camera perspective manually in the pptk viewer. It accepts a list as
        its single argument where the list defines the lookat position (x, y, z) the azimuthal angle, the elevation
        angle, and the distance from the lookat position. This list is returned from the method 'get_perspective()'.
        """
        if self.viewer_is_ready():
            self.viewer.set(lookat=p[0:3], phi=p[3], theta=p[4], r=p[5])

    def select(self, indices=None, highlighted=True, showing=False, classes=None, data=None, intensity=None,
               red=None, green=None, blue=None, compliment=False):
        """
        Return a mask indicating the selected points. Select points based on a number of methods including by
        index, by color, by class, by intensity, or by which points are rendered or highlighted in the viewer.
        If multiple selection methods are used at once, return the intersection of the selections.
        If compliment is True, then return everything EXCEPT the selected points.
        :param indices: List of point indices relative to the entire point cloud.
        :param highlighted: If True, then only grab the points currently highlighted in the viewer.
        :param showing: If True, then only grab points that are currently rendered.
        :param classes: Some list or iterable range from 0 to 31.
        :param data: Some list or iterable range from 0 to 255.
        :param intensity: Some list or iterable range from 0 to 255.
        :param red: Some list or iterable range from 0 to 255.
        :param green: Some list or iterable range from 0 to 255.
        :param blue: Some list or iterable range from 0 to 255.
        :param compliment: If True, return a mask indicating the NON-selected points.
        :return: Boolean mask indicating which points are selected relative to the full set.
        """
        if 'r' not in self.points:
            red, green, blue = None, None, None

        mask = Mask(len(self.points), True)
        cur_mask = Mask(len(self.points), False)
        if indices is not None and len(indices):
            mask.setr(indices)
        if highlighted:
            cur_mask = self.get_highlighted_mask()
            if cur_mask.count():
                mask.intersection(cur_mask.bools)
        if showing:
            mask.intersection(self.showing.bools)
        if classes is not None:
            cur_mask.false()
            for c in classes:
                cur_mask.union(self.points['class'] == c)
            mask.intersection(cur_mask.bools)
        if data is not None:
            cur_mask.false()
            for d in data:
                cur_mask.union(self.points['user_data'] == d)
            mask.intersection(cur_mask.bools)
        if intensity is not None:
            cur_mask.false()
            for i in intensity:
                cur_mask.union(self.points['intensity'] == i)
            mask.intersection(cur_mask.bools)
        if red is not None:
            cur_mask.false()
            for r in red:
                cur_mask.union(self.points['r'] == r)
            mask.intersection(cur_mask.bools)
        if green is not None:
            cur_mask.false()
            for g in green:
                cur_mask.union(self.points['g'] == g)
            mask.intersection(cur_mask.bools)
        if blue is not None:
            cur_mask.false()
            for b in blue:
                cur_mask.union(self.points['b'] == b)
            mask.intersection(cur_mask.bools)
        if compliment:
            mask.compliment()
        return mask

    def classify(self, cls, overwrite=False, mask=None):
        """
        Set the class of the currently selected points to cls. If the class is already set, then only
        overwrite the old value if "overwrite" is True.
        """
        if mask is None:
            mask = self.get_highlighted_mask()
        if overwrite:
            mask.intersection(self.points['class'] > 0)
        self.points.loc[mask.bools, 'class'] = cls
        self.render(showing=True)

    def center(self):
        """
        Shift the origin of the point cloud to its centroid.
        """
        self.points[['x', 'y', 'z']] -= np.average(self.points[['x', 'y', 'z']], axis=0)

    def reset_origin(self):
        """
        Shift the origin of the point cloud to the minimum of the point cloud.
        """
        self.points[['x', 'y', 'z']] -= self.points[['x', 'y', 'z']].values.min(axis=0)

    def subsample(self, n=10000000, percent=1.0):
        """
        Return a random sample of the point cloud.
        """
        threshold = int(percent * len(self.points))
        if n < threshold:
            threshold = n
        if threshold < len(self.points):
            keep = np.zeros(len(self.points), dtype=bool)
            keep[np.random.choice(len(self.points), threshold)] = True
            return keep
        else:
            return np.ones(len(self.points), dtype=bool)

    def add_points(self, points):
        """
        Append a pandas dataframe of points to the current list of points. The pandas DataFrame
        must have 'x' and 'y' columns, and cannot have any abnormal columns in it.
        """
        if not isinstance(points, pd.DataFrame):
            print('Error: points must be in the form of a pandas DataFrame. Cannot append.')
            return
        if 'x' not in points.columns or 'y' not in points.columns:
            print('Error: missing x and/or y column data. Cannot append.')
            return
        for c in points.columns:
            if c not in self.points.columns:
                print('Error: unknown column', c, 'in points. Cannot append.')
                return
        self.points = self.points.append(points)
        self.points = self.points.fillna(0)

    def slice(self, points=None, position=1.75, thickness=0.2, axis=2):
        """
        Take a planar slice of some thickness out of the data. The slice will be axis-aligned.
        :param points: Set of points to take slice from. Default is all currently rendered points.
        :param position: Position along axis to take slice from. Default is 1.75, set for slicing vertical poles above pallets.
        :param thickness: Thickness of slice to take. Default is 0.2 (20 cm).
        :param axis: Axis perpendicular to the slice of data. Must be 0, 1, or 2 (default is 2 (z-axis)).
        """
        if axis == 2:
            str_axis = 'z'
        elif axis == 1:
            str_axis = 'y'
        else:
            str_axis = 'x'
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        mask = points[str_axis] > position
        mask[points[str_axis] > position + thickness] = False
        return mask

    def in_box_2d(self, box, points=None):
        """
        Return a boolean mask indicating which points are within the given 2D bounding box (xy plane)
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y']].values
        keep = points > np.array(box[0])
        keep[points > np.array(box[1])] = False
        return keep.all(axis=1)

    @staticmethod
    def make_box_from_point(point, delta):
        """
        Make a square bounding box with center at the given point and side lengths 2*delta
        """
        point = np.array(point)
        return [point - delta, point + delta]

    def get_points_within(self, delta, point=None, return_mask=False, return_z=False, proportion=1.0):
        """
        Returns all the points within delta of the given point. Z-axis is not considered in distance calculation.
        If point is None, then use the currently highlighted point as the query point. If multiple points are currently
        highlighted, then use their average as the query point.
        """
        if proportion < 1.0:
            choice = np.random.choice(len(self), int(len(self) * proportion))
            points = self.points.iloc[choice][['x', 'y']]
            if return_mask:
                print('From get_points_within: Doesn\'t make sense to return mask when proportion < 1.0')
                return []
        else:
            points = self.points[['x', 'y']]
        if point is None:
            selected = self.get_highlighted_mask()
            point = np.average(self.points.loc[selected.bools][['x', 'y']], axis=0)
        x, y = point[:2]
        keep = np.ones(len(points), dtype=bool)
        keep[points['x'] < x - delta] = False
        keep[points['x'] > x + delta] = False
        keep[points['y'] < y - delta] = False
        keep[points['y'] > y + delta] = False
        if return_mask:
            return keep
        elif return_z:
            return points.loc[keep][['x', 'y', 'z']].values
        else:
            return points.loc[keep][['x', 'y']].values

    def distance_to_line(self, line, point):
        """
        Calculate the distance between a given point and a given line
        :param line: (x1, y1), (x2, y2)
        :param point: (x, y)
        """
        p1, p2 = line
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if not dx and not dy:
            return np.sqrt(np.square(point[0] - p1[0]) + np.square(point[1] - p1[1]))
        num = np.abs(dy * point[0] - dx * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
        den = np.sqrt(np.square(dx) + np.square(dy))
        return num / den

    def get_points_near_line(self, line, delta=0.01):
        """
        Return a boolean mask indicating which points are within delta of the given line
        :param line: (x1, y1), (x2, y2)
        """
        keep = np.zeros(len(self.points), dtype=bool)
        for i, p in self.points:
            if self.distance_to_line(line, p) < delta:
                keep[i] = True
        return np.arange(len(self.points))[keep]

    def rounding_filter(self, points=None, round=0.02):
        """
        This function rounds the point locations to the nearest "round" (default is 2 cm)
        :return: Unique set of rounded points
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        return np.unique(np.round(points / round, decimals=0) * round)

    def radial_filter(self, points=None, threshold=10, radius=0.05):
        """
        This filter checks that each point has at least threshold neighboring points within the given radius.
        A boolean mask is returned indicating which points passed through the filter.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        query = knn.Query()
        query.pptk(points)
        keep = np.ones(len(points), dtype=bool)
        for i, p in enumerate(tqdm(points, desc='Finding neighbors and filtering')):
            neighbors = query.neighbors(p, k=threshold, radius=radius)
            if len(neighbors) < threshold:
                keep[i] = False
        return keep

    def neighbors(self, k=100, highlight=True):
        """
        Find the centroid of the currently highlighted points and return a Mask indicating which points are neighbors.
        :param k: Number of neighbors to find.
        :param highlight: If True, then set the currently highlighted points to the neareest k neighbors. Default True.
        :return: Return a copy of the mask indicating which points are neighbors of the selected point(s).
        """
        mask = self.select(showing=False, highlighted=True)
        if not mask.count() or mask.count() == len(self.points):
            print('No points were selected')
            return

        points = self.points.loc[mask.bools][['x', 'y', 'z']]
        if len(points) == 1:
            query = points
        else:
            query = np.average(points, axis=0)

        if self.index is None:
            self.index = knn.Query()
            self.index.pptk(self.points.loc[self.showing.bools][['x', 'y', 'z']].values)

        neighbors = self.index.neighbors(query, k)
        mask.setr(neighbors)
        if highlight:
            self.highlight(mask)
        return mask

    def plane_filter(self, points=None, mesh=0.06, axis=2):
        """
        This filter returns the number of counts of points in each slice of the point cloud segmented in
        the given dimension (axis).
        :param points: Set of points to run the method on. By default, run the method on all currently rendered points.
        :param mesh: Mesh size for dividing the point cloud along the given axis.
        :param axis: Axis choice should be 0, 1, or 2 for x, y, and z respectively.
        :return: List of per-point scores or counts indicating how many points share the plane with a given point.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        mesh = np.ones(3) * mesh
        if axis == 0:
            mesh[[1, 2]] = 10000000.0
        elif axis == 1:
            mesh[[0, 2]] = 10000000.0
        elif axis == 2:
            mesh[[0, 1]] = 10000000.0

        vg = VoxelGrid(points, mesh)
        return np.array([vg.counts(vg.index(p)) for p in points])

    def color_groups(self, groups, store=False, render=True):
        """
        Given a list of lists of indices, color all the points in a given group one random color. If store=True, then
        store the coloring scheme in the user_data array.
        """
        labels = np.zeros(len(self.points), dtype=int)
        if not len(groups):
            return []
        elif not isinstance(groups[0], list):
            groups = [groups]

        for indices in groups:
            label = np.random.randint(1, 32)
            if len(indices):
                labels[indices] = label

        if render and self.viewer_is_ready():
            self.viewer.attributes(labels[self.showing.bools])
        if store:
            self.points['user_data'] = labels
        else:
            return labels

    def regularize(self, points=None, mesh=0.02):
        """
        Regularize the density of the given points using a voxel grid of given mesh size. Simply removes all points
        except one per voxel, so if mesh=0.01, then only 1 point per cubic centimeter will be kept, and all others
        will be thrown away. The first point found in the voxel is kept.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        vg = VoxelGrid(points, mesh_size=mesh)
        keep = np.zeros(len(points), dtype=bool)
        for indices in vg.indices():
            keep[indices[0]] = True
        return keep

    def get_points_in_bounds(self, bounds, points=None, extra=0.0):
        """
        This function returns a boolean mask indicating which points are contained in the given bounds.
        If extra is non-zero, then the extra amount will be added to the x and y dimensions of the bounds in order
        to grab points slightly inside or outside the boundary.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values
        keep_min = (bounds[0][:2] - extra < points[:, :2]).all(axis=1)
        keep_max = (bounds[1][:2] + extra > points[:, :2]).all(axis=1)
        keep = keep_min
        keep[keep_max == False] = False
        return keep

    def auto_align_bound_box_method(self, tolerance=0.1, max_points=10000, thickness=0.2):
        """
        Automatically align the point cloud with the x and y axes. This method incrementally rotates the point
        cloud and finds its bounding box, then shrinks the bounding box to see how many points become excluded.
        If the point cloud represents a rectangular prism, then when the walls are aligned with the x,y axes, many
        points will fall outside the shrunken bounding box. If the walls are not aligned, then only the corners will
        fall outside of the shrunken bounding box.
        :param tolerance: Align the point cloud to within this tolerance (in degrees)
        :param max_points: Only consider up to max_points points to speed up the calculation
        :param thickness: Thickness of the bounding shell to use
        """
        points = self.points[['x', 'y']].values
        if max_points < len(points):
            subsample = np.random.choice(len(points), max_points)
            points = points[subsample]
        best_cost, best_angle = np.inf, 0.0
        for i in range(int(90. / tolerance)):
            bounds = [points.min(axis=0), points.max(axis=0)]
            bounds[0] += thickness
            bounds[1] -= thickness
            inbox = self.in_box_2d(bounds, points)
            cost = inbox.sum()
            if cost < best_cost:
                best_cost = cost
                best_angle = i * tolerance
            points = self.rotate(points, tolerance)
        if best_angle:
            self.points[['x', 'y']] = self.rotate(degrees=best_angle)
            self.render(showing=True)
        return best_angle

    def auto_align_hough_line_method(self, tolerance=0.1, max_points=100000):
        """
        Automatically align the point cloud with the x and y axes. This function uses a Hough transform to find the
        most dominant line in the point cloud and aligns that line with the nearest axis.
        :param tolerance: Align the point cloud to within the given tolerance (in degrees)
        :param max_points: Only consider at most max_points points to speed up computation
        """
        points = self.points[['x', 'y']].values
        if max_points < len(points):
            subsample = np.random.choice(len(points), max_points)
            points = points[subsample]
        # Perform a rough alignment to calculate alignment +/- 5 degrees
        votes, _, tolerance, center = self.hough_lines(points, theta_precision=5.0, angle_range=90)
        angle = np.degrees(list(votes.keys())[np.argmax(list(votes.values()))][1] * tolerance + center)
        # Perform a fine alignment given the results of the rough alignment
        votes, _, tolerance, center = self.hough_lines(points, theta_precision=tolerance, angle_range=5, theta_center=angle)
        angle = np.degrees(list(votes.keys())[np.argmax(list(votes.values()))][1] * tolerance + center)
        print('rotating', angle, 'degrees')
        self.points[['x', 'y']] = self.rotate(degrees=angle)
        self.render(showing=True)

    @staticmethod
    def hough_lines(points, theta_precision=0.5, angle_range=90, rho_precision=0.02, theta_center=0.0):
        """
        This function implements a hough transform for finding lines. The function takes in the representative points,
        converts them to discretized (rho, theta) points and votes into an accumulator called "votes" which is
        returned from the function. The key in "votes" belonging to the largest value represents the most predominant
        line. The key is a discretized (rho, theta), so the actual values are rho * rho_precision and
        theta * theta_precision + theta_center. Only the angles in range(-angle_range, angle_range) will be considered.
        Theta_precision and theta_center are in radians.
        """
        theta_precision, angle_range, theta_center = np.radians(theta_precision), np.radians(angle_range), np.radians(theta_center)
        n_steps = int(angle_range / theta_precision)
        theta_idx = [i for i in range(-n_steps, n_steps+1)]
        thetas = np.array([idx * theta_precision + theta_center for idx in theta_idx])
        cosines = np.array([np.cos(theta) for theta in thetas])
        tangents = np.array([np.tan(theta) for theta in thetas])

        # Cast a vote for each line that this point passes through in the given range and precision
        def vote(point):
            # min_distance = b / sqrt(m^2 + 1) = (y - tan(theta) * x) * cos(theta)
            rhos = np.array((point[1] - tangents * point[0]) * cosines / rho_precision, dtype=int)
            for rho, idx in zip(rhos, theta_idx):
                votes[(rho, idx)] += 1

        votes = defaultdict(int)
        for point in tqdm(points, desc='Finding lines'):
            vote(point)

        return votes, rho_precision, theta_precision, theta_center

    def hough_circles(self, points, radius=0.05, resolution=0.010):
        """
        This function finds circles of a given radius in the point cloud. It only looks for circles lying in
        xy-planes in the data (no vertical circles).
        """
        # Discretize the points
        points = (points[:, :2] / resolution).astype(int)
        # Define a circle to go around each point
        angles = [np.radians(theta) for theta in range(0, 360)]
        displacements = radius * np.array([np.array((np.cos(theta), np.sin(theta))) for theta in angles])
        # Discretize the circle and only count each discrete location once
        displacements = np.unique((displacements / resolution).astype(int), axis=0)
        # For each point, make a vote for each circle that this point could belong to
        votes = defaultdict(int)
        for point in points:
            for d in displacements:
                votes[tuple(point + d)] += 1
        # In order to get the circle center, find the key corresponding to high voted bin and multiply by resolution
        return votes, radius, resolution

    def hough_squares(self, points, length=0.1, resolution=0.010):
        """
        This function finds squares of a given radius in the point cloud. It only looks for squares lying in
        xy-planes in the data (no vertical squares).
        """
        # Discretize the points
        points = (points[:, :2] / resolution).astype(int)
        # Define a square to go around each point
        n = int(length / resolution / 2.)
        displacements = [np.array((x, n)) for x in range(-n, n+1)]
        displacements += [np.array((x, -n)) for x in range(-n, n+1)]
        displacements += [np.array((n, y)) for y in range(-n+1, n)]
        displacements += [np.array((-n, y)) for y in range(-n+1, n)]
        # For each point, make a vote for each square that this point could belong to
        votes = defaultdict(int)
        for point in points:
            for d in displacements:
                votes[tuple(point + d)] += 1
        # In order to get the square center, find the key corresponding to high voted bin and multiply by resolution
        return votes, length, resolution

    def hough_intersections(self, points, resolution=0.01):
        """
        This function finds the location of two intersecting, non-parallel lines in the point cloud.
        """
        votes, rho_precision, theta_precision, theta_center = self.hough_lines(points, theta_precision=0.1, rho_precision=resolution)
        n = 100
        best = np.argsort(list(votes.values()))[-n:]
        best_keys = np.array(list(votes.keys()))[best]
        dists, angles = best_keys[:, 0] * rho_precision, best_keys[:, 1] * theta_precision + theta_center
        values = np.array(list(votes.values()))[best]
        best_pair, best_score = None, 0
        for i in range(len(best)):
            for j in range(i, len(best)):
                score = (values[i] + values[j]) * abs(np.sin(angles[i] - angles[j]))
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        i, j = best_pair
        theta1, theta2 = angles[i], angles[j]
        rho1, rho2 = dists[i], dists[j]
        m1, m2 = np.tan(theta1 + np.pi / 2.), np.tan(theta2 + np.pi / 2.)
        x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
        x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
        X = ((m1 * x1 - y1) - (m2 * x2 - y2)) / (m1 - m2)
        Y = m1 * (X - x1) + y1
        return X, Y

    def rotate(self, points=None, degrees=0.0, axes=['x', 'y']):
        """
        This function takes in a list of points (or uses the currently rendered points) and returns a set of points that
        have been rotated by 'degrees' degrees about the given axis. By default, axis=2, so the points will rotate
        about the z-axis.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][axes].values

        t = np.radians(degrees)
        rot = np.array(((np.cos(t), -np.sin(t)), (np.sin(t), np.cos(t))))
        return np.dot(points, rot)

    def normals(self, points=None, k=100, r=0.35, render=False):
        """
        This function takes in a set of points (or uses the currently rendered points) and calculates the surface
        normals using pptk built-in functions which use PCA method. The number of neighbors (k) or the distance
        scale (r) can be changed to affect the resolution of the computation. If render=True, then the results
        will be rendered upon completion.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        n = np.abs(pptk.estimate_normals(points, k, r))
        if render and self.viewer_is_ready():
            self.viewer.attributes(n)

        return n

    def curvature(self, points=None, k=100, r=0.35):
        """
        This function takes in a set of points (or uses the currently rendered points) and calculates the surface
        curvature using pptk built-in functions which use PCA method. The number of neighbors (k) or the distance
        scale (r) can be changed to affect the resolution of the computation. If render=True, then the results
        will be rendered upon completion.
        """
        if points is None:
            points = self.points.loc[self.showing.bools][['x', 'y', 'z']].values

        eigens = np.abs(pptk.estimate_normals(points, k, r, output_eigenvalues=True)[0])
        eigens.sort(axis=1)
        return eigens[:, 0] / eigens.sum(axis=1) * 3.0
