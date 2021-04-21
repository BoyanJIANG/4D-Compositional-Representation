import torch
import torch.distributions as dist
import os
from torch import nn
from lib.encoder import encoder_dict, encoder_temporal_dict
from lib.cr import models, training, generation
from lib import data


def get_decoder(cfg, device, c_dim=0):
    ''' Returns a decoder instance.

    Args:
        cfg (yaml config): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    inp_dim = c_dim*2

    if decoder:
        decoder = models.decoder_dict[decoder](
            c_dim=inp_dim,
            **decoder_kwargs).to(device)
    else:
        decoder = None

    return decoder

def get_velocity_field(cfg, device, dim=3, c_dim=0):
    ''' Returns a velocity field instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    '''
    velocity_field = cfg['model']['velocity_field']
    velocity_field_kwargs = cfg['model']['velocity_field_kwargs']

    if velocity_field:
        velocity_field = models.velocity_field_dict[velocity_field](
            in_dim=129, out_dim=128, c_dim=c_dim, **velocity_field_kwargs
        ).to(device)
    else:
        velocity_field = None

    return velocity_field


def get_encoder(cfg, device, dataset=None, c_dim=0):
    ''' Returns an encoder instance.

    If input type if 'idx', the encoder consists of an embedding layer.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dataset (dataset): dataset
        c_dim (int): dimension of conditioned code c
    '''
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    if encoder == 'idx':
        if cfg['model']['learn_embedding']:
            encoder = nn.Sequential(
                nn.Embedding(len(dataset), 128),
                nn.Linear(128, c_dim)).to(device)
        else:
            encoder = nn.Embedding(len(dataset), c_dim).to(device)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, dim=3, **encoder_kwargs).to(device)
    else:
        encoder = None

    return encoder


def get_encoder_temporal(cfg, device, c_dim=0):
    ''' Returns a temporal encoder instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        c_dim (int): dimension of latent conditioned code c
        z_dim (int): dimension of latent code z
    '''
    encoder_temporal = cfg['model']['encoder_temporal']
    encoder_temporal_kwargs = cfg['model']['encoder_temporal_kwargs']
    length_sequence = cfg['data']['length_sequence']

    if encoder_temporal:
        encoder_temporal = encoder_temporal_dict[encoder_temporal](
            c_dim=c_dim, dim=length_sequence*3, **encoder_temporal_kwargs).to(device)
    else:
        encoder_temporal = None

    return encoder_temporal


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Returns a model instance.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
        dataset (dataset): Pytorch dataset
    '''
    # General arguments
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    input_type = cfg['data']['input_type']
    ode_solver = cfg['model']['ode_solver']
    ode_step_size = cfg['model']['ode_step_size']
    use_adjoint = cfg['model']['use_adjoint']
    rtol = cfg['model']['rtol']
    atol = cfg['model']['atol']

    decoder = get_decoder(cfg, device, c_dim)
    encoder = get_encoder(cfg, device, dataset, c_dim=128)
    encoder_motion = get_encoder_temporal(cfg, device, c_dim)
    encoder_identity = get_encoder(cfg, device, dataset, c_dim=128)

    velocity_field = get_velocity_field(cfg, device, dim, c_dim)

    model = models.Compositional4D(
        decoder=decoder, encoder=encoder,
        encoder_motion=encoder_motion, encoder_identity=encoder_identity, vector_field=velocity_field,
        ode_step_size=ode_step_size, use_adjoint=use_adjoint,
        rtol=rtol, atol=atol, ode_solver=ode_solver,
        device=device, input_type=input_type)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): model instance
        optimzer (torch.optim): Pytorch optimizer
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    eval_sample = cfg['training']['eval_sample']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        threshold=threshold
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): model instance
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        padding=cfg['generation']['padding'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        n_time_steps=cfg['generation']['n_time_steps'],
        only_end_time_points=cfg['generation']['only_end_time_points'],
    )

    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution.

    Args:
        cfg (yaml): yaml config
        device (device): Pytorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_transforms(cfg):
    ''' Returns transforms.

    Args:
        cfg (yaml): yaml config
    '''
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']
    transf_pt = data.SubsamplePoints(n_pt)
    transf_pt_val = data.SubsamplePointsSeq(n_pt_eval, random=False)

    return transf_pt, transf_pt_val


def get_data_fields(mode, cfg):
    ''' Returns data fields.

    Args:
        mode (str): mode (train | val | test)å
        cfg (yaml): yaml config
    '''
    fields = {}
    seq_len = cfg['data']['length_sequence']
    p_folder = cfg['data']['points_iou_seq_folder']
    transf_pt, transf_pt_val = get_transforms(cfg)
    unpackbits = cfg['data']['points_unpackbits']
    scale_type = cfg['data']['scale_type']

    pts_iou_field = data.PointsSubseqField

    if mode == 'train':
        fields['points'] = pts_iou_field(p_folder, transform=transf_pt,
                                         seq_len=seq_len,
                                         fixed_time_step=0,
                                         unpackbits=unpackbits,
                                         scale_type=scale_type)
        fields['points_t'] = pts_iou_field(p_folder,
                                           transform=transf_pt,
                                           seq_len=seq_len,
                                           unpackbits=unpackbits,
                                           scale_type=scale_type)
        fields['points_ex'] = pts_iou_field(p_folder, transform=transf_pt,
                                         seq_len=seq_len,
                                         fixed_time_step=0,
                                         unpackbits=unpackbits,
                                         scale_type=scale_type)
        fields['points_t_ex'] = pts_iou_field(p_folder,
                                           transform=transf_pt,
                                           seq_len=seq_len,
                                           unpackbits=unpackbits,
                                           scale_type=scale_type)
    elif mode == 'val':
        fields['points_iou'] = pts_iou_field(p_folder, transform=transf_pt_val,
                                             all_steps=True, seq_len=seq_len,
                                             unpackbits=unpackbits,
                                             scale_type=scale_type)
    return fields
