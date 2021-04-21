import argparse
import trimesh
import numpy as np
import os
import shutil
import glob
import sys
from multiprocessing import Pool
from functools import partial
sys.path.append(os.getcwd())
from lib.utils.libmesh import check_mesh_contains
from smpl_torch_batch import SMPLModel


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--out_folder', type=str,
                    help='Path to save the outputs.')
parser.add_argument('--ext', type=str, default='obj',
                    help='Extensions for meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--pointcloud_folder', type=str, default='pcl_seq',
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')

parser.add_argument('--points_folder', type=str, default='points_seq',
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=0.5,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                    help='Whether to save truth values as bit array.')


def main(args):
    hids = os.listdir(os.path.join(args.in_folder))
    seq_folders = []
    for hid in hids:
        seq_folders.extend(glob.glob(os.path.join(args.in_folder, hid, '*')))
    seq_folders.sort()
    print('Total number of sequences: ', len(seq_folders))

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), seq_folders)
    else:
        for p in seq_folders:
            process_path(p, args)


def process_path(in_path, args):
    smpl_model = SMPLModel(model_path='data/human_dataset/smpl_models/model_300_m.pkl')
    smpl_faces = smpl_model.faces

    identity, motion = in_path.split('/')[-2:]
    model_file = os.path.join(in_path, 'smpl_vers.npy')

    # Export various modalities
    if args.pointcloud_folder is not None:
        export_pointcloud(identity, motion, model_file, smpl_faces, args)

    if args.points_folder is not None:
        export_points(identity, motion, model_file, smpl_faces, args)

    print(in_path)


def get_loc_scale(mesh, args):
    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

    return loc, scale


# Export functions
def export_pointcloud(identity, motion, model_files, smpl_faces, args):
    out_folder = os.path.join(args.out_folder, 'D-FAUST', identity,
                              motion, args.pointcloud_folder)

    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Pointcloud already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)

    # Create out_folder
    os.makedirs(out_folder)

    all_vers = np.load(model_files)
    mesh = trimesh.Trimesh(all_vers[0].squeeze(), smpl_faces.squeeze(), process=False)
    _, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    alpha = np.random.dirichlet((1,)*3, args.pointcloud_size)

    for it, verts in enumerate(all_vers):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.Trimesh(verts.squeeze(), smpl_faces.squeeze(), process=False)
        loc = np.zeros(3)
        scale = np.array([1.])

        vertices = mesh.vertices
        faces = mesh.faces
        v = vertices[faces[face_idx]]
        points = (alpha[:, :, None] * v).sum(axis=1)

        print('Writing pointcloud: %s' % out_file)
        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points = points.astype(dtype)
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)

        np.savez(out_file, points=points, loc=loc, scale=scale)



def export_points(identity, motion, model_files, smpl_faces, args):
    out_folder = os.path.join(args.out_folder, 'D-FAUST', identity,
                              motion, args.points_folder)

    if os.path.exists(out_folder):
        if not args.overwrite:
            print('Points already exist: %s' % out_folder)
            return
        else:
            shutil.rmtree(out_folder)

    # Create out_folder
    os.makedirs(out_folder)

    all_vers = np.load(model_files)

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    for it, verts in enumerate(all_vers):
        out_file = os.path.join(out_folder, '%08d.npz' % it)
        mesh = trimesh.Trimesh(verts.squeeze(), smpl_faces.squeeze(), process=False)
        if not mesh.is_watertight:
            print('Warning: mesh %s is not watertight!')

        loc_self, scale_self = get_loc_scale(mesh, args)
        loc_global = np.array([-0.005493, -0.1888, 0.07587])
        scale_global = np.array([2.338])
        mesh.apply_translation(-loc_global)
        mesh.apply_scale(1/scale_global)

        boxsize = 1 + args.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = boxsize * (points_uniform - 0.5)
        points_uniform = (loc_self + scale_self * points_uniform - loc_global) / scale_global

        points_surface = mesh.sample(n_points_surface)
        points_surface += args.points_sigma * \
            np.random.randn(n_points_surface, 3)
        points = np.concatenate([points_uniform, points_surface], axis=0)

        occupancies = check_mesh_contains(mesh, points)
        print('Writing points: %s' % out_file)

        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points = points.astype(dtype)
        loc = loc_global.astype(dtype)
        scale = scale_global.astype(dtype)

        if args.packbits:
            occupancies = np.packbits(occupancies)

        np.savez(out_file, points=points, occupancies=occupancies,
                 loc=loc, scale=scale)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
