import torch
import numpy as np
from torch.nn import functional as F
from lib.common import compute_iou
from lib.training import BaseTrainer


class Trainer(BaseTrainer):
    r''' Trainer object for ONet 4D.

    Onet 4D is trained with BCE. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state.

    Args:
        model (nn.Module): Onet 4D model
        optimizer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
        threshold (float): threshold value for decision boundary
    '''

    def __init__(self, model, optimizer, device=None, input_type='img', threshold=0.4):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.threshold = threshold

    def train_step(self, data):
        ''' Performs a train step.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (tensor): validation data
        '''
        self.model.eval()
        device = self.device
        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
        batch_size, seq_len, n_pts, _ = inputs.size()
        eval_dict = {}
        loss = 0
        with torch.no_grad():
            # Encode inputs
            c_p, c_m, c_i = self.model.encode_inputs(inputs)

            # IoU
            eval_dict_iou = self.eval_step_iou(data, c_m=c_m, c_p=c_p, c_i=c_i)
            for (k, v) in eval_dict_iou.items():
                eval_dict[k] = v
                loss += eval_dict['iou']

        eval_dict['loss'] = loss.mean().item()
        return eval_dict



    def eval_step_iou(self, data, c_p=None, c_m=None, c_i=None):
        ''' Calculates the IoU for the evaluation step.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        '''
        device = self.device
        threshold = self.threshold
        eval_dict = {}

        pts_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ')
        pts_iou_t = data.get('points_iou.time').to(device)

        batch_size, n_steps, n_pts, dim = pts_iou.shape

        p = pts_iou

        c_i = c_i.unsqueeze(0).repeat(1, n_steps, 1)
        c_p_at_t = self.model.transform_to_t_eval(pts_iou_t[0], p=c_p, c_t=c_m)
        c_s_at_t = torch.cat([c_i, c_p_at_t], -1)

        c_s_at_t = c_s_at_t.view(batch_size * n_steps, c_s_at_t.shape[-1])

        p = p.view(batch_size * n_steps, n_pts, -1)
        occ_iou = occ_iou.view(batch_size * n_steps, n_pts)

        occ_pred = self.model.decode(p, c=c_s_at_t)

        occ_pred = (occ_pred.probs > threshold).cpu().numpy()
        occ_gt = (occ_iou >= 0.5).numpy()
        iou = compute_iou(occ_pred, occ_gt)

        iou = iou.reshape(batch_size, -1).mean(0)

        eval_dict['iou'] = iou.sum() / len(iou)
        for i in range(len(iou)):
            eval_dict['iou_t%d' % i] = iou[i]

        return eval_dict


    def get_loss_recon_t(self, data, c_p=None, c_m=None, c_i=None, is_exchange=None):
        ''' Calculates the reconstruction loss.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        '''
        device = self.device
        if is_exchange:
            p_t = data.get('points_t_ex').to(device)
            occ_t = data.get('points_t_ex.occ').to(device)
            time_val = data.get('points_t_ex.time').to(device)
        else:
            p_t = data.get('points_t').to(device)
            occ_t = data.get('points_t.occ').to(device)
            time_val = data.get('points_t.time').to(device)

        batch_size, n_pts, _ = p_t.shape

        c_p_at_t = self.model.transform_to_t(time_val, p=c_p, c_t=c_m)
        c_s_at_t = torch.cat([c_i, c_p_at_t], 1)

        p = p_t
        logits_pred = self.model.decode(p, c=c_s_at_t).logits

        loss_occ_t = F.binary_cross_entropy_with_logits(
            logits_pred, occ_t.view(batch_size, -1), reduction='none')

        loss_occ_t = loss_occ_t.mean()

        return loss_occ_t


    def get_loss_recon_t0(self, data, c_p=None, c_i=None, is_exchange=None):
        ''' Calculates the reconstruction loss.

        Args:
            data (tensor): training data
            c_t (tensor): temporal conditioned latent code
            z (tensor): latent code
        '''
        if is_exchange:
            p_t0 = data.get('points_ex')
            occ_t0 = data.get('points_ex.occ')
        else:
            p_t0 = data.get('points')
            occ_t0 = data.get('points.occ')

        batch_size, n_pts, _ = p_t0.shape

        device = self.device
        batch_size = p_t0.shape[0]

        c_s_at_t0 = torch.cat([c_i, c_p], 1)

        p = p_t0

        logits_t0 = self.model.decode(p.to(device), c=c_s_at_t0).logits
        loss_occ_t0 = F.binary_cross_entropy_with_logits(
            logits_t0, occ_t0.view(batch_size, -1).to(device),
            reduction='none')

        loss_occ_t0 = loss_occ_t0.mean()

        return loss_occ_t0


    def compute_loss(self, data):
        ''' Calculates the loss.

        Args:
            data (tensor): training data
        '''
        device = self.device
        seq1, seq2 = data
        # Encode inputs
        inputs1 = seq1.get('inputs', torch.empty(1, 1, 0)).to(device)
        inputs2 = seq2.get('inputs', torch.empty(1, 1, 0)).to(device)

        c_p_1, c_m_1, c_i_1 = self.model.encode_inputs(inputs1)
        c_p_2, c_m_2, c_i_2 = self.model.encode_inputs(inputs2)

        is_exchange = np.random.randint(2)

        if is_exchange:
            in_c_i_1 = c_i_2
            in_c_i_2 = c_i_1
        else:
            in_c_i_1 = c_i_1
            in_c_i_2 = c_i_2

        loss_recon_t_1 = self.get_loss_recon_t(seq1, c_m=c_m_1, c_p=c_p_1, c_i=in_c_i_1, is_exchange=is_exchange)
        loss_recon_t0_1 = self.get_loss_recon_t0(seq1, c_p=c_p_1, c_i=in_c_i_1, is_exchange=is_exchange)
        loss_recon_t_2 = self.get_loss_recon_t(seq2, c_m=c_m_2, c_p=c_p_2, c_i=in_c_i_2, is_exchange=is_exchange)
        loss_recon_t0_2 = self.get_loss_recon_t0(seq2, c_p=c_p_2, c_i=in_c_i_2, is_exchange=is_exchange)

        loss_recon_t = (loss_recon_t_1 + loss_recon_t_2) / 2.0
        loss_recon_t0 = (loss_recon_t0_1 + loss_recon_t0_2) / 2.0
        loss = loss_recon_t + loss_recon_t0

        return loss
