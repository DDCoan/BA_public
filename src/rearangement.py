"""
Split data set for the char sequence problem.
Ran only ONCE.
Data: http://archive.ics.uci.edu/ml/datasets/Character+Trajectories
"""
# import numpy as np
# from scipy import io
# import torch
# import random
# # from utils import *
# # import matplotlib.pyplot as plt
#
# np.random.seed(1234)
#
# def normalise(data):
#     ms = np.mean(data, axis=1, keepdims=True)
#     std = np.std(data, axis=1, keepdims=True)
#     return (data - ms) / std
#
#
# def pad_sequences(seqs):
#     """padding with zeros """
#     # type: (ndarray) -> Tuple(ndarray, ndarray)
#     # Get each sequence and pad it
#     seq_lengths = np.array([s.shape[1] for s in seqs])
#     tensors = [torch.from_numpy(s.T) for s in seqs]
#     seqs = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
#     return seqs, seq_lengths
#
#
# MAT = io.loadmat('../dat/UCI_character_trajectory/mixoutALL_shifted.mat')
# # MAT = io.loadmat('/home/yao/ba/dat/UCI_character_trajectory/mixoutALL_shifted.mat')
# CHARLABELS = np.array(MAT['consts']['charlabels'][0, 0][0].tolist()) - 1
# KEYS = np.array(MAT['consts']['key'][0, 0][0].tolist())
# MIXOUT = MAT['mixout'][0]
# SEQS, SEQ_LENGTHS = pad_sequences(MIXOUT)
#
#
# # def visualization(s,l,bsz):
# #     fig, ax = plt.subplots(1, bsz, sharex='col', sharey='row')
# #     start = np.zeros(2)
# #     delta_t = .01
# #     traj = v2trajectory(start, s, delta_t)
# #     for i in range(bsz):
# #         ax[i].plot(traj[i, :, 0], traj[i, :, 1])
# #     print(KEYS[l])
# #     plt.show()
#
#
# def v2trajectory(start, data, delta_t):
#     n = len(data)
#     traj = np.tile(start, (n, 1, 1))
#     for i in range(len(data[0])):
#         new_pos = np.add(traj[:, -1, :], data[:, i, :2] * delta_t)
#         traj = np.append(traj, new_pos[:, None, :], axis=1)
#     return traj[:, 1:, :]
#
#
# #
# trajectory = normalise(v2trajectory(np.zeros(2), SEQS, 0.01))
# seqs_withxy = np.append(SEQS, trajectory, axis=-1)
#
#
# test_split = .1
# N = len(SEQS)
# INDICES = random.sample(range(len(SEQS)), N)
# seqs, charlabels, seq_lengths = seqs_withxy[INDICES], CHARLABELS[INDICES], SEQ_LENGTHS[INDICES]
#
# # split TRAINING and TEST
# idx = np.arange(len(INDICES))
# test_split = int(np.floor(test_split * len(idx)))
# TRAINING_INDICES, TEST_INDICES = idx[test_split:], idx[:test_split]
#
# # split and store training data
# TRAINING_DATA, TRAINING_LABELS, TRAINING_LENGTHS = seqs[TRAINING_INDICES], charlabels[TRAINING_INDICES], seq_lengths[TRAINING_INDICES]
# torch.save(torch.from_numpy(TRAINING_DATA).float(), '../dat/training_data.pt')
# torch.save(torch.from_numpy(TRAINING_LABELS), '../dat/training_labels.pt')
# torch.save(torch.from_numpy(TRAINING_LENGTHS), '../dat/training_lengths.pt')
#
#
# # split and store test data
# TEST_DATA, TEST_LABELS, TEST_LENGTHS = seqs[TEST_INDICES], charlabels[TEST_INDICES], seq_lengths[TEST_INDICES]
# torch.save(torch.from_numpy(TEST_DATA).float(), '../dat/test_data.pt')
# torch.save(torch.from_numpy(TEST_LABELS), '../dat/test_labels.pt')
# torch.save(torch.from_numpy(TEST_LENGTHS), '../dat/test_lengths.pt')
#

