import torch
import torch.nn as nn

import numpy as np
from src.utils import *

class Encoder(nn.Module):
    def __init__(self, Tpred, emb, layer, hid, enc, device='cpu'):
        super(Encoder, self).__init__()
        self.Tpred = Tpred
        self.emb = emb
        self.hid = hid
        self.enc = enc
        self.num_layers = layer
        self.embed_layer = nn.Linear(2, emb)
        self.gru_layer = nn.GRU(emb, hid, num_layers=self.num_layers, batch_first=True)
        self.transform_layer = nn.Linear(hid, enc)

        self.activation = nn.ReLU()

        self.device = device

    def forward(self, x_true):
        x = self.embed_layer(x_true)
        x = self.activation(x)
        
        hidden = torch.zeros(self.num_layers, x_true.size(0), self.hid).to(self.device)
        alpha, hid = self.gru_layer(x[:,0].unsqueeze(1),hidden)
        for i in range(1,self.Tpred-1):
            alpha, hidden = self.gru_layer(x[:,i].unsqueeze(1), hidden)
        
        representation = self.transform_layer(alpha)

        return representation
    

class Decoder(nn.Module):
    def __init__(self, Tpred = 12, enc=4, layer=3, hidden=4, device='cpu'):
        super(Decoder, self).__init__()
        self.num_layers = layer
        self.hid = hidden
        self.Tpred = Tpred
        self.input_layer = nn.Linear(enc, enc)
        self.gru_layer = nn.GRU(enc, self.hid, num_layers=self.num_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hid, 2)

        self.activation = nn.ReLU()
        self.device = device

    def forward(self, representation, inference):
        predictions = torch.tensor([]).to(self.device)

        if inference:
            for i in range(representation.size(1)):
                rep = representation[:,i]
                pred = torch.tensor([]).to(self.device)
                hidden = torch.zeros(self.num_layers, rep.size(0), self.hid).to(self.device)

                for j in range(self.Tpred):
                    rep = self.input_layer(rep)
                    rep = self.activation(rep)
                    output, hidden = self.gru_layer(rep, hidden)
                    output_transformed = self.output_layer(output)
                    pred = torch.cat((pred,output_transformed),dim=-2)
                    rep = hidden[-1,:,:].unsqueeze(1)

                predictions = torch.cat((predictions, pred.unsqueeze(1)), dim=1)

            return predictions

        hidden = torch.zeros(self.num_layers, representation.size(0), self.hid).to(self.device)

        for i in range(self.Tpred):
            representation = self.input_layer(representation)
            representation = self.activation(representation)
            output, hidden = self.gru_layer(representation, hidden)
            output_transformed = self.output_layer(output)
            predictions = torch.cat((predictions,output_transformed),dim=-2)
            representation = hidden[-1,:,:].unsqueeze(1)

        return predictions


class RNN_AE(nn.Module):
    def __init__(self, FLAGS):
        super().__init__()
        self.Tpred = FLAGS.prediction_len
        self.emb = FLAGS.ae_embedding
        self.layer = FLAGS.ae_layers
        self.hidden = FLAGS.ae_hidden
        self.enc = FLAGS.ae_enc
        self.device = FLAGS.device

        self.encoder = Encoder(Tpred=self.Tpred, emb=self.emb, layer=self.layer, hid=self.hidden, enc=self.enc, device=self.device).to(self.device)
        self.decoder = Decoder(Tpred=self.Tpred, enc=self.enc, layer=self.layer, hidden=self.hidden, device=self.device).to(self.device)

        self.loss_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
    
    def compute_loss(self, input, output):
        ade = ((input[:,:,0]-output[:,:,0])**2+(input[:,:,1]-output[:,:,1])**2).sqrt().mean(1)
        return ade

    def encoding(self, y_true):
        return self.encoder(y_true)
    
    def decoding(self, enc, inference=False):
        return self.decoder(enc, inference)