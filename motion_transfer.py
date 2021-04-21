import torch
from torchvision import transforms
import os
import argparse
from lib import config, data
from lib.checkpoints import CheckpointIO


parser = argparse.ArgumentParser(
    description='Motion transfer'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--g', type=str, default='0', help='gpu id')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.g

cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, 'motion_transfer')

# Dataset
connected_samples = cfg['data']['input_pointcloud_corresponding']
transform = transforms.Compose([
    data.SubsamplePointcloudSeq(
        cfg['data']['input_pointcloud_n'],
        connected_samples=connected_samples),
        data.PointcloudNoise(cfg['data']['input_pointcloud_noise'])
])
fields = {
    'inputs': data.PointCloudSubseqField(
        cfg['data']['pointcloud_seq_folder'],
        transform, seq_len=cfg['data']['length_sequence'],
        scale_type=cfg['data']['scale_type'])
}
dataset = data.HumansDataset(dataset_folder=cfg['data']['path'],
                             fields=fields, mode='test', split='test')


# Choose the motion sequence and identity sequence
identity_seq = {'category': 'D-FAUST', 'model': '50002_light_hopping_loose', 'start_idx': 30}
motion_seq = {'category': 'D-FAUST', 'model': '50004_punching', 'start_idx': 60}

inp_id = dataset.get_data_dict(identity_seq)
inp_motion = dataset.get_data_dict(motion_seq)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

model.eval()
meshes, _ = generator.generate_motion_transfer(inp_id, inp_motion)

# Save generated sequence
if not os.path.isdir(generation_dir):
    os.makedirs(generation_dir)
modelname = '%s_%d_to_%s_%d' % (motion_seq['model'],
                                motion_seq['start_idx'],
                                identity_seq['model'],
                                identity_seq['start_idx'])
print('Saving mesh to ', generation_dir)
generator.export(meshes, generation_dir, modelname)
