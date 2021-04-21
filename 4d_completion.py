from lib import config, data
import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import trange


def back_optim(model, generator, data_loader, out_dir, device, time_value, t_idx, latent_size,
               code_std=0.1, num_iterations=500):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    id_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda()
    pose_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda()
    motion_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda()
    id_code.requires_grad = True
    pose_code.requires_grad = True
    motion_code.requires_grad = True

    optimizer = optim.Adam([id_code, pose_code, motion_code], lr=0.03)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print('Seen Frames:')
    print(t_idx)

    with trange(num_iterations, ncols=80) as steps:
        iters = 0
        for _ in steps:
            for batch in data_loader:
                iters += 1

                idx = batch['idx'].item()

                try:
                    model_dict = dataset.get_model_dict(idx)
                except AttributeError:
                    model_dict = {'model': str(idx), 'category': 'n/a'}

                model.eval()
                optimizer.zero_grad()

                pts_iou = batch.get('points')
                occ_iou = batch.get('points.occ')
                pts_iou_t = torch.from_numpy(time_value).to(device)

                batch_size, _, n_pts, dim = pts_iou.shape
                n_steps = pts_iou_t.shape[0]

                p = pts_iou[:, t_idx, :, :].to(device)
                occ_gt = occ_iou[:, t_idx, :].to(device)

                c_i = id_code.unsqueeze(0).repeat(1, n_steps, 1)
                c_p_at_t = model.transform_to_t_eval(pts_iou_t, p=pose_code, c_t=motion_code)
                c_s_at_t = torch.cat([c_i, c_p_at_t], -1)

                c_s_at_t = c_s_at_t.view(batch_size * n_steps, c_s_at_t.shape[-1])

                p = p.view(batch_size * n_steps, n_pts, -1)
                occ_gt = occ_gt.view(batch_size * n_steps, n_pts)

                logits_pred = model.decode(p, c=c_s_at_t).logits

                loss_recons = F.binary_cross_entropy_with_logits(
                    logits_pred, occ_gt.view(n_steps, -1), reduction='none')

                loss_recons = loss_recons.mean()
                loss = loss_recons
                loss.backward()
                steps.set_postfix(Loss=loss.item())
                optimizer.step()
                lr_sche.step()

                if iters % 100 == 0:
                    # -----------visualization-------------
                    out = generator.generate_for_completion(id_code, pose_code, motion_code)
                    try:
                        mesh, stats_dict = out
                    except TypeError:
                        mesh, stats_dict = out, {}

                    # Save files
                    modelname = model_dict['model']
                    start_idx = model_dict.get('start_idx', 0)
                    print('Saving meshes...')
                    generator.export(mesh, out_dir, modelname, start_idx)

                    # -----------save codes-------------
                    print('Saving latent vectors...')
                    torch.save(
                        {"it": iters,
                         "id_code": id_code,
                         "pose_code": pose_code,
                         "motion_code": motion_code,
                         "Observations": t_idx},
                         os.path.join(out_dir, 'latent_vecs_%d.pt' % start_idx)
                    )



if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Conduct backward optimization experiments.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Type of the backward experiment. temporal, spatial or future')
    parser.add_argument('--seq', type=str, default='50026_shake_arms',
                        help='Name of the sequence')
    parser.add_argument('--start_idx', type=int, default=30,
                        help='Start index of the sub-sequence')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of the sub-sequence,'
                             'we set it to 30 for 4D completion and 20 for future prediction.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--g', type=str, default='0', help='gpu id')
    args = parser.parse_args()

    assert args.experiment in ['temporal', 'spatial', 'future']
    if args.experiment == 'future':
        args.seq_length = 20

    os.environ['CUDA_VISIBLE_DEVICES'] = args.g

    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    transf_pt = data.SubsamplePointsSeq(cfg['data']['n_training_points'],  random=True,
                                        spatial_completion=True if args.experiment == 'spatial' else False)
    fields = {
        'points': data.PointsSubseqField(
            cfg['data']['points_iou_seq_folder'], all_steps=True,
            seq_len=args.seq_length,
            unpackbits=cfg['data']['points_unpackbits'],
            transform=transf_pt,
            scale_type=cfg['data']['scale_type'],
            spatial_completion=True if args.experiment == 'spatial' else False),
        'idx': data.IndexField(),
    }

    specific_model = {'seq': args.seq,
                      'start_idx': args.start_idx}

    ################
    out_dir = cfg['training']['out_dir']
    mesh_out_folder = os.path.join(out_dir, args.experiment, args.seq)
    dataset_folder = cfg['data']['path']
    ################

    dataset = data.HumansDataset(dataset_folder, fields, 'test',
                                 specific_model=specific_model)

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1,
        worker_init_fn=data.worker_init_fn,
        shuffle=False)

    model = config.get_model(cfg, device=device)
    model_dir = os.path.join(out_dir, cfg['test']['model_file'])
    print('Loading checkpoint from %s' % model_dir)
    load_dict = torch.load(model_dir)
    model.load_state_dict(load_dict['model'])

    cfg['generation']['n_time_steps'] = args.seq_length
    generator = config.get_generator(model, cfg, device=device)

    times = np.array([i / (args.seq_length - 1) for i in range(args.seq_length)], dtype=np.float32)
    if args.experiment == 'temporal':
        t_idx = np.random.choice(range(args.seq_length), size=args.seq_length // 2, replace=False)
        t_idx.sort()
    elif args.experiment == 'spatial':
        t_idx = np.arange(args.seq_length)
    else:
        t_idx = np.arange(args.seq_length // 2)

    back_optim(model, generator, test_loader, out_dir=mesh_out_folder,
               latent_size=cfg['model']['c_dim'],
               device=device, num_iterations=500,
               time_value=times[t_idx], t_idx=t_idx)
