import os
import torch
import numpy as np
import random
import xarray as xr
from torch.utils.data import Dataset
from utils.sampling import CauchySampler

sampler = CauchySampler(scale=0.2, num_dims=3, min=-1, max=1.)

W_LEN = 64
TS_IDX = 1000
WS = 1
WINDOWS = 961

class SampledTimeDataset(Dataset):

    def __init__(self, data:list):
        self.data = data
        self.N = self.data.shape[0]
    
    def __getitem__(self, idx):
            pos_idx_ref, pos_idx, pos_t_ref, pos_t, corr = self.data[idx, :1], self.data[idx, 1:2], self.data[idx, 2:4], self.data[idx, 4:6], self.data[idx, -1]
            return pos_idx_ref, pos_idx, pos_t_ref, pos_t, corr
    
    def __len__(self):
        return self.N

class TimeDatasetRow(Dataset):

    def __init__(self, data, num_points, grid='random'):
        self.data = data
        self.num_points = num_points
        self.grid = grid
        self.pairs = self.grid_sampler()
        self.N = self.pairs[1].shape[0]
    
    def __getitem__(self, idx):
            pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_ = self.pairs[0][idx, :], self.pairs[1][idx, :], self.pairs[2][idx, :], self.pairs[3][idx, :], self.pairs[4][idx, :], self.pairs[5][idx, :]
            return pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_
    
    def __len__(self):
        return self.N
    
    def grid_sampler(self):
        data_min = np.nanmin(self.data)
        data_max = np.nanmax(self.data)
        input = torch.from_numpy(self.data).unsqueeze(1)
        if self.grid == 'random':
            time_regular = torch.linspace(-1, 1, TS_IDX)
            sampled_t_points_ref = 2 * torch.rand(self.num_points, 1) - 1
            sampled_w_points = torch.zeros((self.num_points, 1))
            sampled_idx_ref, sampled_idx = zip(*[random.sample(range(0, TS_IDX), 2) for z in range(self.num_points)]) 
            sampled_idx_ref = torch.tensor(sampled_idx_ref)
            sampled_tserie_ref = time_regular[sampled_idx_ref]
            sampled_t_w_points_ref = torch.cat([sampled_t_points_ref, sampled_w_points], dim=-1)
            sampled_t_points_ref_2d = torch.cat([sampled_t_w_points_ref, sampled_tserie_ref.unsqueeze(-1)], dim=-1)
            sampled_t_points = sampled_t_points_ref 
            sampled_idx = torch.tensor(sampled_idx)
            sampled_tserie = time_regular[sampled_idx]
            sampled_t_w_points = torch.cat([sampled_t_points, sampled_w_points], dim=-1)
            sampled_t_points_2d = torch.cat([sampled_t_w_points, sampled_tserie.unsqueeze(-1)], dim=-1)
            
        elif self.grid == 'distance':
            sampled_positions_ref, sampled_positions = sampler.sample(self.num_points)
        out_ref, idx_ref = self._grid_sampler(input, sampled_t_points_ref_2d, data_min, data_max)
        out, idx = self._grid_sampler(input, sampled_t_points_2d, data_min, data_max)
        idx_new = torch.logical_and(idx_ref, idx)
        sampled_t_w_points_ref, sampled_t_w_points = sampled_t_w_points_ref[idx_new], sampled_t_w_points[idx_new]
        sampled_idx_ref, sampled_idx = sampled_idx_ref[idx_new], sampled_idx[idx_new]
        out_ref, out = out_ref[idx_new], out[idx_new]
        return sampled_idx_ref.unsqueeze(-1), sampled_idx.unsqueeze(-1), sampled_t_w_points_ref, sampled_t_w_points, out_ref, out

    
    def _grid_sampler(self, input, positions, min, max):
        member_grid = positions.unsqueeze(0).repeat(W_LEN, 1, 1)  
        grid = member_grid.unsqueeze(2).unsqueeze(2)
        output = torch.nn.functional.grid_sample(input.float(), grid, align_corners=True)
        output_squeezed = torch.squeeze(output)
        idx = ~torch.any(output_squeezed.isnan(), dim=0)
        out_norm = (output_squeezed - min) / (max - min)
        out_norm = torch.transpose(out_norm, 0, 1)
        return out_norm, idx


class TimeDatasetShiftAll(Dataset):

    def __init__(self, data, num_points, grid='random'):
        self.data = data
        self.num_points = num_points
        self.grid = grid
        self.pairs = self.grid_sampler()
        self.N = self.pairs[1].shape[0]
    
    def __getitem__(self, idx):
            pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_ = self.pairs[0][idx, :], self.pairs[1][idx, :], self.pairs[2][idx, :], self.pairs[3][idx, :], self.pairs[4][idx, :], self.pairs[5][idx, :]
            return pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_
    
    def __len__(self):
        return self.N
    
    def grid_sampler(self):
        data_min = np.nanmin(self.data)
        data_max = np.nanmax(self.data)
        input = torch.from_numpy(self.data).unsqueeze(1)
        if self.grid == 'random':
            time_regular = torch.linspace(-1, 1, TS_IDX)
            sampled_t_points_ref = 2 * torch.rand(self.num_points, 1) - 1
            sampled_w_points = torch.zeros((self.num_points, 1))
            sampled_idx_ref, sampled_idx = zip(*[random.sample(range(0, TS_IDX), 2) for z in range(self.num_points)]) 
            sampled_idx_ref = torch.tensor(sampled_idx_ref)
            sampled_tserie_ref = time_regular[sampled_idx_ref]
            sampled_t_w_points_ref = torch.cat([sampled_t_points_ref, sampled_w_points], dim=-1)
            sampled_t_points_ref_2d = torch.cat([sampled_t_w_points_ref, sampled_tserie_ref.unsqueeze(-1)], dim=-1)
            sampled_t_points = 2 * torch.rand(self.num_points, 1) - 1 
            sampled_idx = torch.tensor(sampled_idx)
            sampled_tserie = time_regular[sampled_idx]
            sampled_t_w_points = torch.cat([sampled_t_points, sampled_w_points], dim=-1)
            sampled_t_points_2d = torch.cat([sampled_t_w_points, sampled_tserie.unsqueeze(-1)], dim=-1)
            
        elif self.grid == 'distance':
            sampled_positions_ref, sampled_positions = sampler.sample(self.num_points)
        out_ref, idx_ref = self._grid_sampler(input, sampled_t_points_ref_2d, data_min, data_max)
        out, idx = self._grid_sampler(input, sampled_t_points_2d, data_min, data_max)
        idx_new = torch.logical_and(idx_ref, idx)
        sampled_t_w_points_ref, sampled_t_w_points = sampled_t_w_points_ref[idx_new], sampled_t_w_points[idx_new]
        sampled_idx_ref, sampled_idx = sampled_idx_ref[idx_new], sampled_idx[idx_new]
        out_ref, out = out_ref[idx_new], out[idx_new]
        return sampled_idx_ref.unsqueeze(-1), sampled_idx.unsqueeze(-1), sampled_t_w_points_ref, sampled_t_w_points, out_ref, out

    
    def _grid_sampler(self, input, positions, min, max):
        member_grid = positions.unsqueeze(0).repeat(W_LEN, 1, 1)  
        grid = member_grid.unsqueeze(2).unsqueeze(2)
        output = torch.nn.functional.grid_sample(input.float(), grid, align_corners=True)
        output_squeezed = torch.squeeze(output)
        idx = ~torch.any(output_squeezed.isnan(), dim=0)
        out_norm = (output_squeezed - min) / (max - min)
        out_norm = torch.transpose(out_norm, 0, 1)
        return out_norm, idx

class TimeDatasetShift(Dataset):

    def __init__(self, data, num_points, shift=64, grid='random'):
        self.data = data
        self.num_points = num_points
        self.shift = shift
        self.grid = grid
        self.pairs = self.grid_sampler()
        self.N = self.pairs[1].shape[0]
    
    def __getitem__(self, idx):
            pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_ = self.pairs[0][idx, :], self.pairs[1][idx, :], self.pairs[2][idx, :], self.pairs[3][idx, :], self.pairs[4][idx, :], self.pairs[5][idx, :]
            return pos_idx_ref, pos_idx, pos_t_ref, pos_t, in_ref, in_
    
    def __len__(self):
        return self.N
    
    def grid_sampler(self):
        data_min = np.nanmin(self.data)
        data_max = np.nanmax(self.data)
        input = torch.from_numpy(self.data).unsqueeze(1)
        if self.grid == 'random':
            time_regular = torch.linspace(-1, 1, TS_IDX)
            shift_scaled = (2.0 / (WINDOWS - 1.0)) * self.shift
            sampled_t_points_ref = 2 * torch.rand(self.num_points, 1) -1.0
            t_points = sampled_t_points_ref + 2 * shift_scaled * torch.rand(self.num_points, 1) - shift_scaled
            # t_points = t_points.clamp_(-1, 1)

            # sampled_t_points_ref = 2 * torch.rand(self.num_points, 2) - 1 # 3 refers to 0->x(lon), 1->y(lat), 2->z(lev)
            # sampled_t_points_ref = 2 * torch.rand(self.num_points, 1) - 1
            sampled_w_points = torch.zeros((self.num_points, 1))
            sampled_idx_ref, sampled_idx = zip(*[random.sample(range(0, TS_IDX), 2) for z in range(self.num_points)]) 
            sampled_idx_ref = torch.tensor(sampled_idx_ref)
            # sampled_idx_ref = torch.randint(0, LEVELS, (self.num_points,))
            sampled_tserie_ref = time_regular[sampled_idx_ref]
            sampled_t_w_points_ref = torch.cat([sampled_t_points_ref, sampled_w_points], dim=-1)
            sampled_t_points_ref_2d = torch.cat([sampled_t_w_points_ref, sampled_tserie_ref.unsqueeze(-1)], dim=-1)
            sampled_t_points = torch.clamp(t_points, -1.0, 1.0)
            # sampled_t_points = sampled_t_points_ref 
            sampled_idx = torch.tensor(sampled_idx)
            # sampled_idx = torch.randint(0, LEVELS, (self.num_points,))
            sampled_tserie = time_regular[sampled_idx]
            sampled_t_w_points = torch.cat([sampled_t_points, sampled_w_points], dim=-1)
            sampled_t_points_2d = torch.cat([sampled_t_w_points, sampled_tserie.unsqueeze(-1)], dim=-1)
            
        elif self.grid == 'distance':
            sampled_positions_ref, sampled_positions = sampler.sample(self.num_points)
        out_ref, idx_ref = self._grid_sampler(input, sampled_t_points_ref_2d, data_min, data_max)
        out, idx = self._grid_sampler(input, sampled_t_points_2d, data_min, data_max)
        idx_new = torch.logical_and(idx_ref, idx)
        sampled_t_w_points_ref, sampled_t_w_points = sampled_t_w_points_ref[idx_new], sampled_t_w_points[idx_new]
        sampled_idx_ref, sampled_idx = sampled_idx_ref[idx_new], sampled_idx[idx_new]
        out_ref, out = out_ref[idx_new], out[idx_new]
        return sampled_idx_ref.unsqueeze(-1), sampled_idx.unsqueeze(-1), sampled_t_w_points_ref, sampled_t_w_points, out_ref, out

    
    def _grid_sampler(self, input, positions, min, max):
        member_grid = positions.unsqueeze(0).repeat(W_LEN, 1, 1)  # add member dim
        grid = member_grid.unsqueeze(2).unsqueeze(2)
        output = torch.nn.functional.grid_sample(input.float(), grid, align_corners=True)
        output_squeezed = torch.squeeze(output)
        idx = ~torch.any(output_squeezed.isnan(), dim=0)
        out_norm = (output_squeezed - min) / (max - min)
        out_norm = torch.transpose(out_norm, 0, 1)
        return out_norm, idx

