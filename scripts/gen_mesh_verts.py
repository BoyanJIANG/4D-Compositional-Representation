import os
import torch
import pickle
import numpy as np
from smpl_torch_batch import SMPLModel
from tqdm import tqdm


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

male = ['50002', '50007', '50009', '50026', '50027']
female = ['50004', '50020', '50022', '50025']
hid = male + female

model_m = SMPLModel(model_path='data/human_dataset/smpl_models/model_300_m.pkl', device=device)
model_f = SMPLModel(model_path='data/human_dataset/smpl_models/model_300_f.pkl', device=device)

with open('data/human_dataset/test/D-FAUST/train.lst', 'r') as f:
    all_seq = f.read().split('\n')
all_seq = list(filter(lambda x: len(x) > 0, all_seq))
all_id = pickle.load(open('data/human_dataset/all_betas.pkl', 'rb'))

out_path = 'data/human_dataset/all_train_mesh'
for identity in hid:
    print('human id: ', identity)
    model = model_m if identity in male else model_f
    for seq in tqdm(all_seq):
        code_path = os.path.join('data/human_dataset/smpl_params', seq)
        seq_len = len(os.listdir(code_path))
        pose_codes = [np.load(os.path.join(code_path, str(i).zfill(8) + '.npz'))['pose']
                      for i in range(seq_len)]
        pose_codes = np.array(pose_codes)  # seq_len, 72
        id_code = np.tile(all_id[identity][None, ...], [pose_codes.shape[0], 1])  # seq_len, 300
        beta = torch.from_numpy(id_code).double().to(device)
        pose = torch.from_numpy(pose_codes).double().to(device)
        trans = torch.zeros(pose_codes.shape[0], 3).double().to(device)
        verts = model(beta, pose, trans)  # seq_len, 6890, 3

        save_dir = os.path.join(out_path, identity, seq)
        os.makedirs(save_dir, exist_ok=True)
        verts = np.around(verts.float().detach().cpu(), 6)
        np.save(os.path.join(save_dir, 'smpl_vers.npy'), verts)

print('Done!')