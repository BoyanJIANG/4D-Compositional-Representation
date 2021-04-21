import torch
import torch.nn as nn
from torch import distributions as dist
from lib.cr.models import decoder, velocity_field
from lib.utils.torchdiffeq.torchdiffeq import odeint, odeint_adjoint



decoder_dict = {
    'cbatchnorm': decoder.DecoderCBatchNorm,
}

velocity_field_dict = {
    'concat': velocity_field.VelocityField,
}


class Compositional4D(nn.Module):
    ''' Networks for 4D compositional representation.

    Args:
        decoder (nn.Module): Decoder model
        encoder_latent (nn.Module): Latent encoder model
        encoder_temporal (nn.Module): Temporal encoder model
        p0_z (dist): Prior distribution over latent codes z
        device (device): Pytorch device
        input_type (str): Input type
    '''

    def __init__(
        self, decoder, encoder_latent=None, vector_field=None,
            encoder=None, encoder_motion=None, encoder_identity=None, ode_step_size=None, use_adjoint=False,
            rtol=0.001, atol=0.00001, ode_solver='dopri5',
            device=None, input_type=None, **kwargs):
        super().__init__()

        self.device = device
        self.input_type = input_type

        self.decoder = decoder
        self.encoder_latent = encoder_latent
        self.encoder = encoder
        self.encoder_motion = encoder_motion
        self.encoder_identity = encoder_identity
        self.vector_field = vector_field

        self.rtol = rtol
        self.atol = atol
        self.ode_solver = ode_solver

        if use_adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

        self.ode_options = {}
        if ode_step_size:
            self.ode_options['step_size'] = ode_step_size


    def decode(self, p, c=None, **kwargs):
        ''' Returns occupancy values for the points p at time step t.

        Args:
            p (tensor): points of dimension 4
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial, whereas for ONet 4D, this is c_temporal)
        '''
        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def encode_inputs(self, inputs):
        ''' Returns encoded latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        '''
        c_p = self.encoder(inputs[:, 0, :])
        c_m = self.encoder_motion(inputs)
        c_i = self.encoder_identity(inputs[:, 0, :])

        return c_p, c_m, c_i

    def transform_to_t(self, t, p, c_t=None):
        '''  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, t_order = self.eval_ODE(t, p, c_t=c_t, return_start=True)
        batch_size = len(t_order)
        p_out = p_out[torch.arange(batch_size), t_order]

        return p_out

    def transform_to_t_eval(self, t, p, c_t=None):
        '''  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, _ = self.eval_ODE(t, p, c_t=c_t, return_start=(0 in t))

        return p_out


    # ######################################################
    # #### ODE related functions and helper functions #### #

    def eval_ODE(self, t, p, c_t=None, t_batch=None, invert=False,
                 return_start=False):
        ''' Evaluates the ODE for points p and time values t.

        Args:
            t (tensor): time values
            p (tensor): points tensor
            c_t (tensor): latent conditioned temporal code
            z (tensor): latent code
            t_batch (tensor): helper time tensor for batch processing of points
                with different time values when going backwards
            invert (bool): whether to invert the velocity field (used for
                batch processing of points with different time values)
            return_start (bool): whether to return the start points
        '''
        c_dim = c_t.shape[-1]
        p_dim = p.shape[-1]

        t_steps_eval, t_order = self.return_time_steps(t)
        if len(t_steps_eval) == 1:
            return p.unsqueeze(1), t_order

        f_options = {'T_batch': t_batch, 'invert': invert}
        p = self.concat_vf_input(p, c=c_t)
        s = self.odeint(
            self.vector_field, p, t_steps_eval,
            method=self.ode_solver, rtol=self.rtol, atol=self.atol,
            options=self.ode_options, f_options=f_options)

        p_out = self.disentangle_vf_output(
            s, p_dim=p_dim, c_dim=c_dim, return_start=return_start)

        return p_out, t_order


    def return_time_steps(self, t):
        ''' Returns time steps for the ODE Solver.
        The time steps are ordered, duplicates are removed, and time 0
        is added for the start.

        Args:
            t (tensor): time values
        '''
        device = self.device
        t_steps_eval, t_order = torch.unique(
            torch.cat([torch.zeros(1).to(device), t]), sorted=True,
            return_inverse=True)
        return t_steps_eval, t_order[1:]

    def disentangle_vf_output(self, v_out, p_dim=3, c_dim=None,
                              return_start=False):
        ''' Disentangles the output of the velocity field.

        The inputs and outputs for / of the velocity network are concatenated
        to be able to use the adjoint method.

        Args:
            v_out (tensor): output of the velocity field
            p_dim (int): points dimension
            c_dim (int): dimension of conditioned code c
            return_start (bool): whether to return start points
        '''

        n_steps, batch_size, _ = v_out.shape

        if c_dim is not None and c_dim != 0:
            v_out = v_out[:, :, :-c_dim]

        v_out = v_out.contiguous().view(n_steps, batch_size, p_dim)

        if not return_start:
            v_out = v_out[1:]

        v_out = v_out.transpose(0, 1)

        return v_out

    def concat_vf_input(self, p, c=None):
        ''' Concatenate points p and latent code c to use it as input for ODE Solver.

        p of size (B x T x dim) and c of size (B x c_dim) and z of size
        (B x z_dim) is concatenated to obtain a tensor of size
        (B x (T*dim) + c_dim + z_dim).

        This is done to be able to use to the adjont method for obtaining
        gradients.

        Args:
            p (tensor): points tensor
            c (tensor): latent conditioned code c
        '''
        batch_size = p.shape[0]
        p_out = p.contiguous().view(batch_size, -1)
        if c is not None and c.shape[-1] != 0:
            assert (c.shape[0] == batch_size)
            c = c.contiguous().view(batch_size, -1)
            p_out = torch.cat([p_out, c], dim=-1)

        return p_out