import logging
import os
import math

import numpy as np
import random
import copy

from scipy import stats

import torch
from torch.utils.data import Dataset

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    def __init__(self, FLAGS, test=False, delim='\t'):
        super(TrajectoryDataset, self).__init__()

        self.data_dir = FLAGS.dataset_folder
        self.obs_len = FLAGS.observation_len
        self.pred_len = FLAGS.prediction_len
        self.seq_len = self.obs_len + self.pred_len
        self.cond_ped = FLAGS.conditioning_peds+1
        self.skip = FLAGS.skip
        self.closest = FLAGS.closest
        self.min_radius = FLAGS.min_radius
        self.augmentation = FLAGS.augmentation
        self.delim = delim

        all_dataset_files = os.listdir(self.data_dir)
        all_files = all_dataset_files
        if FLAGS.loo_file:
            if test:
                all_files = FLAGS.loo_file
            else:
                all_files = [x for x in all_dataset_files if x not in FLAGS.loo_file]
        files = [os.path.join(self.data_dir, _path) for _path in all_files]

        seq_list = []

        for path in files:
            data = read_file(path, delim)
            frames = np.unique(data[:,0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame==data[:,0],:])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            # loop on all the sequences
            for idx in range(0, num_sequences*self.skip+1, self.skip):
                # collect all data from the frame idx to the frame idx+sequence
                curr_seq_data = np.concatenate(frame_data[idx:idx+self.seq_len],axis=0)
                # collect peds in the frames considered
                peds_in_curr_seq = np.unique(curr_seq_data[:,1])

                # loop on all the pedestrian in the current sequence
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # init sequence
                    curr_seq = np.zeros((self.cond_ped,2,self.seq_len))
                    num_peds_considered = 0

                    # get data of the pedestrian ped_id 
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # get first and last frames of the sequence
                    first_frame = curr_ped_seq[0, 0]
                    last_frame = curr_ped_seq[-1, 0]
                    pad_front = frames.index(first_frame) - idx
                    pad_end = frames.index(last_frame) - idx + 1
                    if (pad_end - pad_front != self.seq_len) and not(test):
                        continue

                    if (pad_end - pad_front < 10) and test:
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq

                    _idx = num_peds_considered

                    if test:
                        for points in range(len(curr_ped_seq[1])):
                            curr_seq[_idx, :, points] = curr_ped_seq[:, points]
                    else:
                        curr_seq[_idx, :] = curr_ped_seq

                    if np.sqrt(np.sum(np.power((curr_seq[...,-1]-curr_seq[...,0]),2))) >= self.min_radius:
                        if self.cond_ped-1 != 0:
                            num_peds_considered += 1

                            # get all other peds in the scene
                            other_peds_in_scene = np.delete(peds_in_curr_seq, np.where(peds_in_curr_seq == ped_id),axis=0)

                            if self.closest:
                                dist_i = []
                                for i in other_peds_in_scene:
                                    i_frames = curr_seq_data[curr_seq_data[:,1]==i][:,0]
                                    i_coordinates = curr_seq_data[np.isin(curr_seq_data[:,0], i_frames) & (curr_seq_data[:,1]==i)][:,2:4]
                                    ped_coordinates = curr_seq_data[np.isin(curr_seq_data[:,0], i_frames) & (curr_seq_data[:,1]==ped_id)][:,2:4]
                                    if ped_coordinates.shape[0] > i_coordinates.shape[0]:
                                        dist = np.mean(np.linalg.norm(i_coordinates-ped_coordinates[:i_coordinates.shape[0]], axis=1))
                                    elif ped_coordinates.shape[0] < i_coordinates.shape[0]:
                                        dist = np.mean(np.linalg.norm(i_coordinates[:ped_coordinates.shape[0]]-ped_coordinates, axis=1))
                                    else:
                                        dist = np.mean(np.linalg.norm(i_coordinates-ped_coordinates, axis=1))
                                    dist_i.append(dist/len(i_frames))
                                    
                                dist_i = np.array(dist_i)
                                index = np.argsort(dist_i)
                            else:
                                index = list(range(0,len(other_peds_in_scene)))
                                random.shuffle(index)

                            for _, other_ped_id in enumerate(other_peds_in_scene[index]):
                                ped_distance = int((curr_seq_data[curr_seq_data[:,1]==other_ped_id][0,0]-first_frame)/10)
                                other_ped_seq = curr_seq_data[curr_seq_data[:, 1] == other_ped_id, :][:(self.obs_len-ped_distance)]
                                other_ped_seq = np.around(other_ped_seq, decimals=4)
                                if len(other_ped_seq) <= self.obs_len and len(other_ped_seq) > 0:
                                    if other_ped_seq[0, 0] == first_frame:
                                        other_ped_seq = np.transpose(other_ped_seq[:, 2:])
                                        other_ped_seq = other_ped_seq

                                        _idx = num_peds_considered
                                        curr_seq[_idx,:,:other_ped_seq.shape[1]] = other_ped_seq
                                        num_peds_considered += 1
                                    else:
                                        other_ped_seq = np.transpose(other_ped_seq[:, 2:])
                                        other_ped_seq = other_ped_seq

                                        _idx = num_peds_considered
                                        curr_seq[_idx,:,(self.obs_len-other_ped_seq.shape[1]):self.obs_len] = other_ped_seq
                                        num_peds_considered += 1

                                if num_peds_considered >= self.cond_ped:
                                    break

                                
                        seq_list.append(curr_seq)
                        if self.augmentation and not(test):
                            s = stats.truncnorm.rvs((0.3-1)/0.5, (1.7-1)/0.5, loc=1, scale=0.5)

                            mean_pos = (curr_seq[...,0]+curr_seq[...,-1])/2
                            centered_seq = copy.deepcopy(curr_seq)
                            centered_seq[:,0] = [centered_seq[i,0]-mean_pos[i,0] for i in range(len(mean_pos))]
                            centered_seq[:,1] = [centered_seq[i,1]-mean_pos[i,1] for i in range(len(mean_pos))]
                            centered_seq *= s
                            centered_seq[:,0] = [centered_seq[i,0]+mean_pos[i,0] for i in range(len(mean_pos))]
                            centered_seq[:,1] = [centered_seq[i,1]+mean_pos[i,1] for i in range(len(mean_pos))]
                            seq_list.append(centered_seq)

        seq_list = np.array(seq_list)
        self.num_seq = len(seq_list)

        self.obs_traj = torch.from_numpy(seq_list[:,:,:,:self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(np.expand_dims(seq_list[:,0,:,self.obs_len:],axis=1)).type(torch.float)
        print("Done")

    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, index):
        return self.obs_traj[index].transpose(2,1), self.pred_traj[index].transpose(2,1)
