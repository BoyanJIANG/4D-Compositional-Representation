import torch
import os
import numpy as np
from lib.utils.io import save_mesh
from trimesh.exchange.export import export_mesh
import time
from lib.utils.onet_generator import Generator3D as Generator3DONet


class Generator3D(object):
    '''  Generator class for Occupancy Networks 4D.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        device (device): pytorch device
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        simplify_nfaces (int): number of faces the mesh should be simplified to
        n_time_steps (int): number of time steps to generate
        only_ent_time_points (bool): whether to only generate end points
    '''

    def __init__(self, model, device=None, points_batch_size=100000,
                 threshold=0.5, refinement_step=0,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1,
                 sample=False, simplify_nfaces=None, n_time_steps=17,
                 only_end_time_points=False, **kwargs):
        self.n_time_steps = n_time_steps
        self.only_end_time_points = only_end_time_points
        self.onet_generator = Generator3DONet(
            model, device=device,
            points_batch_size=points_batch_size,
            threshold=threshold,
            resolution0=resolution0,
            upsampling_steps=upsampling_steps,padding=padding)

    def generate_mesh_t0(self, c_t=None, data=None, stats_dict={}):
        ''' Generates mesh at first time step.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        '''
        # t = torch.tensor([0.]).view(1, 1).to(self.onet_generator.device)
        # kwargs = {'t': t}
        mesh = self.onet_generator.generate_from_latent(c_t, stats_dict=stats_dict)
        return mesh

    def get_time_steps(self):
        ''' Return time steps values.
        '''
        n_steps = self.n_time_steps
        device = self.onet_generator.device

        if self.only_end_time_points:
            t = torch.tensor([0., 1.]).to(device)
        else:
            t = (torch.arange(1, n_steps).float() / (n_steps - 1)).to(device)

        return t

    def generate_meshes_t(self, c_t=None, data=None, stats_dict={}):
        ''' Generates meshes at time steps > 0.

        Args:
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
            data (dict): data dictionary
            stats_dict (dict): statistics dictionary
        '''
        t = self.get_time_steps()
        meshes = []
        for i, t_v in enumerate(t):
            # kwargs = {'t': t_v.view(1, 1)}
            stats_dict_i = {}
            mesh = self.onet_generator.generate_from_latent(c_t[0, i:i+1], stats_dict=stats_dict_i)
            meshes.append(mesh)
            for k, v in stats_dict_i.items():
                stats_dict[k] += v

        return meshes

    def export_mesh(self, mesh, model_folder, modelname, start_idx=0, n_id=1, out_format='off'):
        ''' Exports a mesh.

        Args:
            mesh(trimesh): mesh to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            n_id (int): number of mesh in the sequence (e.g. 1 -> start)
        '''

        if out_format == 'obj':
            out_path = os.path.join(
                model_folder, '%s_%04d_%04d.obj' % (modelname, start_idx, n_id))
            export_mesh(mesh, out_path)
        else:
            out_path = os.path.join(
                model_folder, '%s_%04d_%04d.off' % (modelname, start_idx, n_id))
            save_mesh(mesh, out_path)
        return out_path

    def export_meshes_t(self, meshes, model_folder, modelname, start_idx=0,
                        start_id_seq=2):
        ''' Exports meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        '''
        out_files = []
        for i, m in enumerate(meshes):
            out_file = self.export_mesh(
                m, model_folder, modelname, start_idx, n_id=start_id_seq + i)
            out_files.append(out_file)

        return out_files

    def export(self, meshes, mesh_dir, modelname, start_idx=0, start_id_seq=1):
        ''' Exports a list of meshes.

        Args:
            meshes (list): list of meshes to export
            model_folder (str): model folder
            model_name (str): name of the model
            start_idx (int): start id of sequence
            start_id_seq (int): start number of first mesh in the sequence
        '''
        model_folder = os.path.join(mesh_dir, modelname, '%05d' % start_idx)
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        return self.export_meshes_t(
            meshes, model_folder, modelname, start_idx=0, start_id_seq=1)

    def generate(self, data):
        ''' Generates meshes for input data.

        Args:
            data (dict): data dictionary
        '''
        self.onet_generator.model.eval()
        stats_dict = {}
        device = self.onet_generator.device
        # Encode inputs
        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        meshes = []
        with torch.no_grad():
            c_p, c_m, c_i = self.onet_generator.model.encode_inputs(inputs)

            # Generate and save first mesh
            c_s_at_t0 = torch.cat([c_i, c_p], 1)

            mesh_t0 = self.generate_mesh_t0(c_s_at_t0, data, stats_dict=stats_dict)
            meshes.append(mesh_t0)

            # Generate and save later time steps
            t = self.get_time_steps()

            c_p_at_t = self.onet_generator.model.transform_to_t_eval(t, p=c_p, c_t=c_m)
            c_s_at_t = torch.cat([c_i.unsqueeze(0).repeat(1, self.n_time_steps - 1, 1), c_p_at_t], -1)

            meshes_t = self.generate_meshes_t(c_t=c_s_at_t, data=data, stats_dict=stats_dict)
            meshes.extend(meshes_t)

        return meshes, stats_dict


    def generate_for_completion(self, c_i, c_p, c_m):
        self.onet_generator.model.eval()
        stats_dict = {}

        meshes = []
        with torch.no_grad():
            c_s_at_t0 = torch.cat([c_i, c_p], 1)

            mesh_t0 = self.generate_mesh_t0(c_s_at_t0, stats_dict=stats_dict)
            meshes.append(mesh_t0)

            # Generate and save later time steps
            t = self.get_time_steps()

            c_p_at_t = self.onet_generator.model.transform_to_t_eval(t, p=c_p, c_t=c_m)
            c_s_at_t = torch.cat([c_i.unsqueeze(0).repeat(1, self.n_time_steps - 1, 1), c_p_at_t], -1)

            meshes_t = self.generate_meshes_t( c_t=c_s_at_t, stats_dict=stats_dict)
            meshes.extend(meshes_t)

        return meshes, stats_dict


    def generate_motion_transfer(self, inp_id, inp_motion):
        self.onet_generator.model.eval()
        stats_dict = {}
        device = self.onet_generator.device
        # Encode inputs
        inp_id = torch.from_numpy(inp_id['inputs']).unsqueeze(0).to(device)
        inp_motion = torch.from_numpy(inp_motion['inputs']).unsqueeze(0).to(device)

        meshes = []
        with torch.no_grad():
            c_i = self.onet_generator.model.encoder_identity(inp_id[:, 0, :])
            c_p = self.onet_generator.model.encoder(inp_motion[:, 0, :])
            c_m = self.onet_generator.model.encoder_motion(inp_motion)

            # Generate and save first mesh
            c_s_at_t0 = torch.cat([c_i, c_p], 1)

            mesh_t0 = self.generate_mesh_t0(c_s_at_t0, stats_dict=stats_dict)
            meshes.append(mesh_t0)

            # Generate and save later time steps
            t = self.get_time_steps()

            c_p_at_t = self.onet_generator.model.transform_to_t_eval(t, p=c_p, c_t=c_m)
            c_s_at_t = torch.cat([c_i.unsqueeze(0).repeat(1, self.n_time_steps - 1, 1), c_p_at_t], -1)

            meshes_t = self.generate_meshes_t(c_t=c_s_at_t, stats_dict=stats_dict)
            meshes.extend(meshes_t)

        return meshes, stats_dict
