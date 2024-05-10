import torch
from torch import nn
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InputProcessing(nn.Module):

    def __init__(self, fourier_module, 
                        latent_feature_module, members, fourier_encoding=True, parametric_encoding= True):
        super(InputProcessing, self).__init__()
        self.fourier_module = fourier_module
        self.latent_feature_module = latent_feature_module
        self.members = members
        self.fourier_encoding = fourier_encoding
        self.parametric_encoding = parametric_encoding

    def forward(self, inputs, pos):
        positions = pos.unsqueeze(1).repeat(1, self.members, 1)
        inputs = inputs.unsqueeze(-1)
        if self.fourier_encoding:
            if self.parametric_encoding:
                encoding = self.fourier_module(pos)
                encoding = encoding.unsqueeze(1).repeat(1, self.members, 1)
                trainable_encoding = self.latent_feature_module(pos)
                trainable_encoding = trainable_encoding.unsqueeze(1).repeat(1, self.members, 1)
                inputs_list = [trainable_encoding, encoding.to(device), inputs, positions]
            else:
                encoding = self.fourier_module(pos)
                encoding = encoding.unsqueeze(1).repeat(1, self.members, 1)
                inputs_list = [encoding, inputs, positions]
        else:
            inputs_list = [inputs, positions]
        inputs = torch.cat(inputs_list, dim=-1)
        return inputs

class InputProcessingTime(nn.Module):

    def __init__(self, fourier_feature_module, latent_feature_module, embedding_module,
                        parametric_encoding=True, fourier_encoding=True, embedding_encoding=True, mul=True):
        super(InputProcessingTime, self).__init__()
        self.fourier_feature_module = fourier_feature_module
        self.embedding_module = embedding_module
        self.latent_feature_module = latent_feature_module
        self.fourier_encoding = fourier_encoding
        self.embedding_encoding = embedding_encoding
        self.parametric_encoding = parametric_encoding
        self.mul = mul

    def forward(self, pos_idx, pos_t):
        fourier_encoding = self.fourier_feature_module(pos_t)
        embedding_encoding = self.embedding_module(pos_idx.int()).squeeze()
        trainable_encoding = self.latent_feature_module(pos_t)
        if self.fourier_encoding:
            t_encoding = torch.cat([fourier_encoding.to(device), trainable_encoding], dim=-1)
        else:
            t_encoding = trainable_encoding
        
        if self.mul:
            pos_encodings = embedding_encoding * t_encoding
        else:
            pos_encodings = torch.cat([embedding_encoding, t_encoding], dim=-1)
        
        inputs_list = [pos_idx, pos_t , pos_encodings]
        inputs = torch.cat(inputs_list, dim=-1)
        
        return inputs