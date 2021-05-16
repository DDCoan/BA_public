"""
Customized dataset for char sequence problem, using remixed data
Original data: http://archive.ics.uci.edu/ml/datasets/Character+Trajectories
"""
import torch.utils


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, seq_lengths):
        # seq_lengths : orginal lengths of samples, needed for pack_padded_sequence
        self.data = seqs
        self.labels = labels
        self.lengths = seq_lengths

    def __getitem__(self, index):

        seq = self.data[index]
        label = self.labels[index]
        length = self.lengths[index]

        return seq, label, length

    def __len__(self):
        return len(self.labels)

