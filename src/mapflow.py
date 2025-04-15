import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from src.spline_flow import NeuralSplineFlow, FCN

import matplotlib.pyplot as plt
from src.autoencoder import RNN_AE
from einops import repeat, rearrange
from src.utils import *

class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, FLAGS, nin, nout, nemb, es=16, hs=16, nl=3, device=0):
        super().__init__()
        self.FLAGS = FLAGS

        self.activation = nn.ReLU()

        self.nl = nl
        self.hs = hs

        self.steps = FLAGS.observation_len
        if FLAGS.rel_pos:
            self.steps = FLAGS.observation_len - 1

        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nemb)

        self.pos_enc_agents = nn.Parameter(torch.randn(self.FLAGS.conditioning_peds+1),requires_grad=True)
        self.pos_enc_time = nn.Parameter(torch.randn(self.steps),requires_grad=True)

        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        size = x.size(0)

        if self.FLAGS.pos_emb_enabled:
            pos_enc_agents = repeat(self.pos_enc_agents, 'n -> b n t p', b=x.size(0), t=self.steps, p=2)
            pos_enc_time = repeat(self.pos_enc_time, 't -> b n t p', b=x.size(0), n=(self.FLAGS.conditioning_peds+1), p=2)
            pos_enc = pos_enc_time + pos_enc_agents

            x = x + pos_enc
    
        x = self.embedding(x)
        x = self.activation(x)
        
        cat = []
        for j in range(x.size(1)):
            hidden = torch.zeros(self.nl, size, self.hs).to(self.device)
            alpha, hidden = self.gru(x[:,j,0].unsqueeze(1), hidden)
            for i in range(1, self.steps):
                alpha, hidden = self.gru(x[:,j,i].unsqueeze(1), hidden)
            
            x_out = self.output_layer(alpha)

            cat.append(x_out)
        x = torch.cat(cat, dim=-1)

        return x, hidden

class MapFlow(nn.Module):

    def __init__(self, FLAGS):
        super().__init__()
        self.pred_steps = FLAGS.prediction_len
        self.output_size = FLAGS.prediction_len * 2
        self.alpha = FLAGS.alpha
        self.beta = FLAGS.beta
        self.gamma = FLAGS.gamma
        self.B = FLAGS.B
        self.rel_coords = FLAGS.rel_pos
        self.norm_rotation = FLAGS.norm_rotation
        self.training = FLAGS.train
        self.ae = FLAGS.ae_enabled

        # core modules
        self.obs_encoding_size = FLAGS.encoding_size*(FLAGS.conditioning_peds+1)
        self.obs_encoding_singleagent = FLAGS.encoding_size
        self.obs_encoder = RNN(FLAGS, nin=2, nout=self.obs_encoding_size, nemb=self.obs_encoding_singleagent, device=FLAGS.device)
        self.autoencoder = RNN_AE(FLAGS)
        if self.ae:
            self.output_size = FLAGS.ae_enc
        self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size, n_layers=FLAGS.num_layers, K=FLAGS.K,
                                    B=self.B, hidden_dim=[FLAGS.hidden_dim, FLAGS.hidden_dim, FLAGS.hidden_dim, FLAGS.hidden_dim, FLAGS.hidden_dim], device=FLAGS.device)

        self.l_weights = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # move model to device
        self.device = FLAGS.device

    def _encode_conditionals(self, x):
        # encode observed trajectory
        x_in = x
        if self.rel_coords:
            x_in = x[:,:,1:] - x[:,:,:-1] # convert to relative coords
        x_enc, hidden = self.obs_encoder(x_in) # encode relative histories
        x_enc = x_enc[:,-1]
        x_enc_context = x_enc
        return x_enc_context

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[:,1:] = (y_rel[:,1:] - y_rel[:,:-1]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 

    def _rotate(self, x, x_t, angles_rad, inference=False):
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        if inference:
            x_center = x - x_t # translate
            x_vals, y_vals = x_center[:,:,0], x_center[:,:,1]
            new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
            new_y_vals = s * x_vals + c * y_vals # _rotate y
            x_center[:,:,0] = new_x_vals
            x_center[:,:,1] = new_y_vals
            return x_center + x_t # translate back
        
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        x_center = x - x_t.unsqueeze(dim=1) # translate
        x_vals, y_vals = x_center[:,:,:,0], x_center[:,:,:,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_center[:,:,:,0] = new_x_vals
        x_center[:,:,:,1] = new_y_vals

        return x_center + x_t.unsqueeze(dim=1) # translate back
    
    def _normalize_rotation(self, x, y_true=None):
        x_t = x[:,0,-1:,:]

        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[:,0,-1] - x[:,0,-2]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[:,1], x_t_rel[:,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad, inference=False)
            y_true = y_true.squeeze(1)
            return x, y_true, rot_angles_rad # inverse

        return x, rot_angles_rad # forward pass

    def _inverse(self, y_true, x):
        if self.norm_rotation:
            x, y_true, angle = self._normalize_rotation(x, y_true)
        else:
            angle = None
            y_true = y_true.squeeze(1)
        x_t = x[:,0,-1:,:]
        x_enc = self._encode_conditionals(x) # history encoding
        y_rel = self._abs_to_rel(y_true, x_t)
        y_rel_flat = torch.flatten(y_rel, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        # CHECK IF AUTOENCODER ENABLED
        if self.ae:
            y_rel = y_rel_flat.view_as(y_rel)
            y_rel_flat = self.autoencoder.encoding(y_rel)
        
        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det, y_rel_flat, x_t, angle

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x, z=None):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x) # history encoding
            x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size)
            n_total = n * x.size(0)
            if self.ae:
                output_shape = (x.size(0), n, 1, self.output_size)
            else:
                output_shape = (x.size(0), n, self.pred_steps, 2) # predict n trajectories input

            # sample and compute likelihoods
            if z is None:
                z = torch.randn([n_total, self.output_size]).to(self.device)

            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            x_t = x[...,0,-1:,:].unsqueeze(dim=1).repeat(1, n, 1, 1)
            
            # DECODING WITH AUTOENCODER IF ENABLED
            if self.ae:
                samples_rel = self.autoencoder.decoding(samples_rel, inference=True)

            samples_abs = self._rel_to_abs(samples_rel, x_t)

            # invert rotation normalization
            if self.norm_rotation:
                x_t_all = x[...,0,-1,:]
                for i in range(len(samples_abs)):
                    pred_trajs = samples_abs[i]
                    samples_abs[i] = self._rotate(pred_trajs, x_t_all[i], -1 * rot_angles_rad[i].repeat(pred_trajs.size(0)),inference=True)

            a = z.clone()
            return samples_abs, log_probs, torch.prod(a.view((x.size(0), 20, -1)),axis=-1)

    def log_prob(self, y_true, x):
        z, log_abs_jacobian_det, y_enc, x_t, angle = self._inverse(y_true, x)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det, normal.log_prob(z).mean(1), y_enc, x_t, angle
    