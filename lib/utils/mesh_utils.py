import numpy as np
import torch
import trimesh
from trimesh.exchange.export import export_mesh


# -------------------------For .obj mesh----------------------------
def save_mesh_trimesh(save_path, verts, faces):
    mesh = trimesh.Trimesh(verts, faces, process=False)
    export_mesh(mesh, save_path)


def save_mesh_np(save_path, verts, faces=None, colors=None):
    if colors is None:
        xyz = np.hstack([np.full([verts.shape[0], 1], 'v'), verts])
        np.savetxt(save_path, xyz, fmt='%s')
    else:
        assert verts.shape[0] == colors.shape[0]
        xyzrgb = np.hstack([np.full([verts.shape[0], 1], 'v'), verts, colors])
        np.savetxt(save_path, xyzrgb, fmt='%s')
    if faces is not None:
        if faces.min == 0:
            faces += 1
        faces = faces.astype(str)
        faces = np.hstack([np.full([faces.shape[0], 1], 'f'), faces])
        with open(save_path, 'a') as f:
            np.savetxt(f, faces, fmt='%s')


def get_mesh_loc_scale(mesh, apply_scale=False):
    ''' Loads and scales a mesh.

    Args:
        mesh_path (trimesh): trimesh
        loc (tuple): location
        scale (float): scaling factor
    '''

    # Compute location and scale
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()

    if apply_scale:
        mesh.apply_translation(-loc)
        mesh.apply_scale(1/scale)

    return mesh, loc, scale, bbox


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
# -------------------------------------------------------------------------



# -------------------------For .off mesh----------------------------
def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces
        # are  all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', \
                'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', \
                'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', \
                    'found empty vertex index: %s (%s)' \
                    % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, \
                'face should have %d vertices but as %d (%s)' \
                % (face[0], len(face) - 1, file)
            assert face[0] == 3, \
                'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, \
                    'vertex %d (of %d vertices) does not exist (%s)' \
                    % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces


def save_off_mesh(mesh, out_file, digits=10, face_colors=None):
    digits = int(digits)
    # prepend a 3 (face count) to each face
    if face_colors is None:
        faces_stacked = np.column_stack((
            np.ones(len(mesh.faces)) * 3, mesh.faces)).astype(np.int64)
    else:
        mesh.visual.face_colors = face_colors
        assert (mesh.visual.face_colors.shape[0] == mesh.faces.shape[0])
        faces_stacked = np.column_stack((
            np.ones(len(mesh.faces)) * 3, mesh.faces,
            mesh.visual.face_colors[:, :3])).astype(np.int64)
    export = 'OFF\n'
    # the header is vertex count, face count, edge number
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += array_to_string(
        mesh.vertices, col_delim=' ', row_delim='\n', digits=digits) + '\n'
    export += array_to_string(faces_stacked, col_delim=' ', row_delim='\n')

    with open(out_file, 'w') as f:
        f.write(export)

    return mesh


def load_off_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        str_file = f.read().split('\n')
        n_vertices, n_faces, _ = list(
            map(lambda x: int(x), str_file[1].split(' ')))
        str_file = str_file[2:]  # Remove first 2 lines

        v = [l.split(' ') for l in str_file[:n_vertices]]
        f = [l.split(' ') for l in str_file[n_vertices:]]

    v = np.array(v).astype(np.float32)
    f = np.array(f).astype(np.uint64)[:, 1:4]

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    return mesh
# -------------------------------------------------------------------------



# -------------------------For .ply point cloud----------------------------
import os
from plyfile import PlyElement, PlyData
from trimesh.util import array_to_string

def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)


def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices
# -------------------------------------------------------------------------
