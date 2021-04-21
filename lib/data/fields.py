import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from lib.data.core import Field
from lib.common import random_crop_occ


class IndexField(Field):
    ''' Basic index field.'''
    # def load(self, model_path, idx, category):

    def load(self, model_path, idx, start_idx=0, dataset_folder=None, **kwargs):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            start_idx (int): id of sequence start
            dataset_folder (str): dataset folder
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True


class PointsSubseqField(Field):
    ''' Points subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        all_steps (bool): whether to return all time steps
        fixed_time_step (int): if and which fixed time step to use
        unpackbits (bool): whether to unpack bits
        scale_type (str, optional): Specifies the type of transformation to apply to the point cloud:
        ``'cr'`` | ``'oflow'``. ``'cr'``: transform the point cloud to align with the output,
        ``'oflow'``: scale the point cloud w.r.t. the first point cloud of the sequence
        spatial_completion (bool): whether to remove some points for 4D spatial completion experiment
    '''

    def __init__(self, folder_name, transform=None, seq_len=17,
                 all_steps=False, fixed_time_step=None, unpackbits=False,
                 scale_type=None, spatial_completion=False, **kwargs):
        self.folder_name = folder_name
        self.transform = transform
        self.seq_len = seq_len
        self.all_steps = all_steps
        self.sample_padding = 0.1
        self.fixed_time_step = fixed_time_step
        self.unpackbits = unpackbits
        self.scale_type = scale_type
        self.spatial_completion = spatial_completion

        if scale_type is not None:
            assert scale_type in ['oflow', 'cr']

    def get_loc_scale(self, mesh):
        ''' Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        '''
        bbox = mesh.bounding_box.bounds

        # Compute location and scale with padding of 0.1
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - self.sample_padding)

        return loc, scale

    def normalize_mesh(self, mesh, loc, scale):
        ''' Normalize mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        '''
        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        return mesh

    def load_files(self, model_path, start_idx):
        ''' Loads the model files.

        Args:
            model_path (str): path to model
            start_idx (int): id of sequence start
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.npz'))
        files.sort()
        files = files[start_idx:start_idx+self.seq_len]

        return files

    def load_all_steps(self, files, loc0, scale0, loc_global, scale_global, dataset_folder):
        ''' Loads data for all steps.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        '''
        p_list = []
        o_list = []
        t_list = []
        for i, f in enumerate(files):
            points_dict = np.load(f)
            
            # Load points
            points = points_dict['points']
            if (points.dtype == np.float16):
                # break symmetry (nec. for some version)
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            occupancies = points_dict['occupancies']
            if self.unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)
            loc = points_dict['loc'].astype(np.float32)
            scale = points_dict['scale'].astype(np.float32)

            model_id, _, frame_id = f.split('/')[-3:]

            # Remove some points for 4D spatial completion experiment
            if self.spatial_completion:
                data_folder = os.path.join(dataset_folder, 'test', 'D-FAUST', model_id)
                mask_folder = os.path.join(dataset_folder, 'spatial_mask', model_id)
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                mask_file = os.path.join(mask_folder, frame_id.replace('.npz', '.npy'))
                if os.path.exists(mask_file):
                    mask = np.load(mask_file)
                else:
                    pcl = np.load(os.path.join(data_folder, 'pcl_seq', frame_id))['points']
                    mask, _, _ = random_crop_occ(points, pcl)
                    np.save(mask_file, mask)

                points = points[mask, :]
                occupancies = occupancies[mask]

            if self.scale_type is not None:
                # Transform to loc0, scale0
                if self.scale_type == 'oflow':
                    points = (loc + scale * points - loc0) / scale0

                # Align the testing data of the original D-FAUST with the output of our model
                if self.scale_type == 'cr':
                    trans = np.load(os.path.join(dataset_folder, 'smpl_params', model_id, frame_id))['trans']
                    loc -= trans
                    points = (loc + scale * points - loc_global) / scale_global

            points = points.astype(np.float32)
            time = np.array(i / (self.seq_len - 1), dtype=np.float32)

            p_list.append(points)
            o_list.append(occupancies)
            t_list.append(time)

        if not self.spatial_completion:
            data = {
                None: np.stack(p_list),
                'occ': np.stack(o_list),
                'time': np.stack(t_list),
            }
        else:
            data = {
                None: p_list,
                'occ': o_list,
                'time': np.stack(t_list),
            }

        return data

    def load_single_step(self, files, points_dict, loc0, scale0):
        ''' Loads data for a single step.

        Args:
            files (list): list of files
            points_dict (dict): points dictionary for first step of sequence
            loc0 (tuple): location of first time step mesh
            scale0 (float): scale of first time step mesh
        '''
        if self.fixed_time_step is None:
            # Random time step
            time_step = np.random.choice(self.seq_len)
        else:
            time_step = int(self.fixed_time_step)

        if time_step != 0:
            points_dict = np.load(files[time_step])

        # Load points
        points = points_dict['points'].astype(np.float32)
        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        if self.scale_type == 'oflow':
            loc = points_dict['loc'].astype(np.float32)
            scale = points_dict['scale'].astype(np.float32)
            # Transform to loc0, scale0
            points = (loc + scale * points - loc0) / scale0

        if self.seq_len > 1:
            time = np.array(
                time_step / (self.seq_len - 1), dtype=np.float32)
        else:
            time = np.array([1], dtype=np.float32)

        data = {
            None: points,
            'occ': occupancies,
            'time': time,
        }
        return data

    def load(self, model_path, idx, c_idx=None, start_idx=0, dataset_folder=None, **kwargs):
        ''' Loads the points subsequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            start_idx (int): id of sequence start
            dataset_folder (str): dataset folder
        '''
        files = self.load_files(model_path, start_idx)
        # Load loc and scale from t_0, we use the global loc and scale calculated from the whole training set
        points_dict = np.load(files[0])
        loc0 = points_dict['loc'].astype(np.float32)
        scale0 = points_dict['scale'].astype(np.float32)

        loc_global = np.array([-0.005493, -0.1888, 0.07587]).astype(np.float32)
        scale_global = 2.338

        if self.all_steps:
            data = self.load_all_steps(files, loc0, scale0, loc_global, scale_global, dataset_folder)
        else:
            data = self.load_single_step(files, points_dict, loc0, scale0)

        if self.transform is not None:
            data = self.transform(data)
        return data


class PointCloudSubseqField(Field):
    ''' Point cloud subsequence field class.

    Args:
        folder_name (str): points folder name
        transform (transform): transform
        seq_len (int): length of sequence
        only_end_points (bool): whether to only return end points
        scale_type (str, optional): Specifies the type of transformation to apply to the input point cloud:
        ``'cr'`` | ``'oflow'``. ``'cr'``: transform the point cloud the original scale and location of SMPL model,
        ``'oflow'``: scale the point cloud w.r.t. the first point cloud of the sequence
    '''

    def __init__(self, folder_name, transform=None, seq_len=17,
                 only_end_points=False, scale_type=None, eval_mode=False):
        self.folder_name = folder_name
        self.transform = transform
        self.seq_len = seq_len
        self.only_end_points = only_end_points
        self.scale_type = scale_type
        self.eval_mode = eval_mode

        if scale_type is not None:
            assert scale_type in ['oflow', 'cr']

    def return_loc_scale(self, mesh):
        ''' Returns location and scale of mesh.

        Args:
            mesh (trimesh): mesh
        '''
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - 0)
        return loc, scale

    def apply_normalization(self, mesh, loc, scale):
        ''' Normalizes the mesh.

        Args:
            mesh (trimesh): mesh
            loc (tuple): location for normalization
            scale (float): scale for normalization
        '''
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)
        return mesh

    def load_files(self, model_path, start_idx):
        ''' Loads the model files.

        Args:
            model_path (str): path to model
            start_idx (int): id of sequence start
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.npz'))
        files.sort()
        files = files[start_idx:start_idx+self.seq_len]

        if self.only_end_points:
            files = [files[0], files[-1]]

        return files

    def load_single_file(self, file_path):
        ''' Loads a single file.

        Args:
            file_path (str): file path
        '''
        pointcloud_dict = np.load(file_path)
        points = pointcloud_dict['points'].astype(np.float32)
        loc = pointcloud_dict['loc'].astype(np.float32)
        scale = pointcloud_dict['scale'].astype(np.float32)

        return points, loc, scale

    def get_time_values(self):
        ''' Returns the time values.
        '''
        if self.seq_len > 1:
            time = \
                np.array([i/(self.seq_len - 1) for i in range(self.seq_len)],
                         dtype=np.float32)
        else:
            time = np.array([1]).astype(np.float32)
        return time

    def load(self, model_path, idx, c_idx=None, start_idx=0, dataset_folder=None, **kwargs):
        ''' Loads the point cloud sequence field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            c_idx (int): index of category
            start_idx (int): id of sequence start
            dataset_folder (str): dataset folder
        '''
        pc_seq = []
        # Get file paths
        files = self.load_files(model_path, start_idx)
        # Load first pcl file
        _, loc0, scale0 = self.load_single_file(files[0])
        loc_global = np.array([-0.005493, -0.1888, 0.07587]).astype(np.float32)
        scale_global = 2.338
        for f in files:
            points, loc, scale = self.load_single_file(f)

            if self.scale_type is not None:
                # Transform mesh to loc0 / scale0
                if self.scale_type == 'oflow':
                    points = (loc + scale * points - loc0) / scale0

                # Transform to original scale and location of SMPL model
                if self.scale_type == 'cr':
                    points = loc + scale * points
                    model_id, _, frame_id = f.split('/')[-3:]
                    trans = np.load(os.path.join(dataset_folder, 'smpl_params', model_id, frame_id))['trans']
                    points = points - trans

                    # Only for evaluation, align the output with the testing data in D-FAUST
                    if self.eval_mode:
                        points = (points - loc_global) / scale_global

            pc_seq.append(points)

        data = {
            None: np.stack(pc_seq),
            'time': self.get_time_values(),
        }

        if self.transform is not None:
            data = self.transform(data)
        return data