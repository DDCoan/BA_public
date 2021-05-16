"""
Training models, including simple RNN, LSTM based on BPTT and eprop1-variants
"""
import torch.nn as nn
import torch.nn.functional as F
from custom_lstm import *
from utils_eprop import *
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_hidden(batch_size, hidden_size):
    h0 = torch.zeros(batch_size, hidden_size, device=device)
    c0 = torch.zeros(batch_size, hidden_size, device=device)
    return h0, c0


###############################################################################
# BPTT for character trajectory problem
###############################################################################
class LstmBpttCharAllTimesteps(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, batch_first=True):
        super(LstmBpttCharAllTimesteps, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inp, inp_lengths):
        if self.batch_first:
            batch_size, seq_len, _ = inp.size()
        else:
            seq_len, batch_size, _ = inp.size()

        # pack padded seq
        pack = torch.nn.utils.rnn.pack_padded_sequence(inp, inp_lengths, batch_first=self.batch_first,
                                                       enforce_sorted=False)

        lstm_output, _ = self.lstm(pack)  # if (h0, c0) not provided, h0 and c0 default to zero

        # undo the packing operation
        unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        unpack = unpack.contiguous()  # [B x T x H]

        # extract the results of last time step
        unpack = torch.cat([y[:l] for (y, l) in zip(unpack, inp_lengths)])  # only last time step

        tag_space = self.linear(unpack)  # B*T,O -> CrossEntropyLoss
        return tag_space


class LstmBpttCharLastTimestep(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, batch_first=True):
        super(LstmBpttCharLastTimestep, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inp, inp_lengths):
        if self.batch_first:
            batch_size, seq_len, _ = inp.size()
        else:
            seq_len, batch_size, _ = inp.size()

        # pack padded seq
        pack = torch.nn.utils.rnn.pack_padded_sequence(inp, inp_lengths, batch_first=self.batch_first,
                                                       enforce_sorted=False)

        lstm_output, _ = self.lstm(pack)  # if (h0, c0) not provided, h0 and c0 default to zero

        # undo the packing operation
        unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        unpack = unpack.contiguous()  # [T x B   x H]

        # extract the results of last time step
        unpack = torch.stack([y[l - 1] for (y, l) in zip(unpack, inp_lengths)])  # only last time step

        tag_space = self.linear(unpack)  # T,B,O -> CrossEntropyLoss
        return tag_space


###############################################################################
# BPTT for sinus problem
###############################################################################

class LstmBpttSin(torch.nn.Module):
    def __init__(self, nhid=3):
        super(LstmBpttSin, self).__init__()
        self.nhid = nhid
        self.lstm = nn.LSTM(1, nhid, batch_first=False)
        self.linear = nn.Linear(nhid, 1, bias=False)

    def forward(self, input, hidden):
        lstm_outputs, hidden = self.lstm(input, hidden)
        ys = self.linear(lstm_outputs)
        return ys, hidden  # , lstm_outputs # lstm outputs only for visualisation


###############################################################################
# E-prop 1 models
###############################################################################
class Eprop1(nn.Module):
    def __init__(self, hidden_size, output_size, feedback_type='symmetric_feedback'):
        super(Eprop1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback_type = feedback_type
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        # initialize random feedback matrix
        stdv = 1. / math.sqrt(self.hidden_size)
        B_random = torch.empty((self.output_size, self.hidden_size), device=device).uniform_(-stdv, stdv)
        self.B_random = B_random

    def symmetric_feedback(self):
        return self.linear.weight.clone()

    def random_feedback(self):
        return self.B_random.clone()


###############################################################################
# E-prop 1 for char and sinus (depending on choice of methods)
# Labels are feed into net at each time step t
###############################################################################


class LstmEprop1AllTimesteps(Eprop1):
    def __init__(self, LstmLayer, input_size, hidden_size, output_size, batch_first=True,
                 feedback_type='symmetric_feedback'):
        # LstmLayer in [LstmNet, LstmC]
        super(LstmEprop1AllTimesteps, self).__init__(hidden_size, output_size, feedback_type)
        self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        self.lstm = LstmLayer(LSTMCell, input_size, hidden_size)
        # self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.batch_first = batch_first
        # self.feedback_type = feedback_type

    # when training, always start with (h0,c0) = zeros, so no need for 'state' as parameter
    def forward(self, inp):
        if self.batch_first:
            inp = inp.transpose(0, 1)
        batch_size = inp.size(1)

        state = init_hidden(batch_size, self.hidden_size)
        lstm_outputs, state, ets, caches = self.lstm(inp, state)  # lstm output : [seq x B x H]
        # output layer
        ys = self.linear(lstm_outputs).view(-1, self.output_size)  # tag space: [seq * B x Out] ~ torch.repeat
        return ys, lstm_outputs, state, ets, caches

    def forward_eval(self, input, state):
        if self.batch_first:
            input = input.transpose(0, 1)
        lstm_outputs, state = self.lstm.forward_eval(input, state)
        ys = self.linear(lstm_outputs).view(-1, self.output_size)  # tag space: [seq * B x Out]
        return ys, state

    def backward_mse(self, ys, lstm_outputs, caches, targets, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        dy = 2 * (ys - targets)[:, :, None]  # S x B x O  # should * 2 in pytorch
        dwout = torch.matmul(dy[:, :, :, None], lstm_outputs[:, :, None, :])
        dls = torch.matmul(dy, self.linear.weight.clone())  # [S x B x O] @ [O x H] -> [S x B x H]
        grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh, grad_h = self.lstm.backward(inputs, dls, caches)

        # assign values to gradients
        # use double mean because in pytorch mse--reduction = 'mean
        self.linear.weight.grad = torch.mean(dwout, dim=(0, 1))
        self.lstm.cell.weight_ih.grad = torch.mean(grad_W_ih, dim=(0, 1))
        self.lstm.cell.weight_hh.grad = torch.mean(grad_W_hh, dim=(0, 1))
        self.lstm.cell.bias_ih.grad = torch.mean(grad_b_ih, dim=(0, 1))
        self.lstm.cell.bias_hh.grad = torch.mean(grad_b_hh, dim=(0, 1))
        return grad_h

    def eprop_mse(self, ys, lstm_outputs, ets, target):
        seq = ys.size(0)
        dEdy = 2 * (ys - target)[:, :, None]  # seq x batch=1 x out=1

        # case: random or symmetric feedback
        W_linear = getattr(self, self.feedback_type)()  # [out x hidden]

        pEph = torch.matmul(dEdy, W_linear)  # [seq x 1 x out] x [out x hidden] - > [seq x 1 x 3]
        pEpW = torch.matmul(dEdy[:, :, :, None],
                            lstm_outputs[:, :, None,
                            :])  # [seq x batch x out x 1] x [seq x batch x 1 x hidden] -> [seq x 1 x 1 x 3]
        # set grad for lstm
        # * .01 because in MSE 'reduction' = 'mean
        grad_wih, grad_whh, grad_bih, grad_bhh = compute_grad_eprop(pEph, ets)
        self.lstm.cell.weight_ih.grad, self.lstm.cell.weight_hh.grad, \
        self.lstm.cell.bias_ih.grad, self.lstm.cell.bias_hh.grad = grad_wih / seq, grad_whh / seq, \
                                                                   grad_bih / seq, grad_bhh / seq
        self.linear.weight.grad = torch.mean(pEpW, dim=(0, 1))  # mean corresponds bptt
        return pEph  # only for ets plotting

    def eprop_cross_entropy(self, ys, lstm_outputs, ets, labels_seqs):
        """
        :param ys: [seq * B, O]
        :param lstm_outputs: [seq, B, H ]
        :param ets: Tuple
        :param labels_seqs: [seq * B, 1] only the correct class, not one-hot encoded, torch.repeat
        :return:
        """
        tag_scores = F.log_softmax(ys, dim=-1)
        seq, batch_size, _ = lstm_outputs.size()
        idx = (torch.arange(self.output_size)).repeat(seq * batch_size, 1).to(device)  # [seq * B x O]
        labels_repeated = torch.repeat_interleave(labels_seqs[:, None], self.output_size, dim=1)  # [seq * B x O]

        # to generate one-hot encoded vector
        dEdy = -(labels_repeated == idx).float() + torch.exp(
            tag_scores)  # [seq * batch x out] , derivative w.r.t. tag scores
        dEdy = dEdy.reshape(seq, batch_size, self.output_size)

        W_linear = getattr(self, self.feedback_type)()

        pEph = torch.matmul(dEdy, W_linear)  # [seq x B x O] @ [O x H] - > [seq x B x H]
        # partial E, W_linear
        pEpW = torch.matmul(dEdy[:, :, :, None],
                            lstm_outputs[:, :, None, :])  # [seq x B x O x 1] @ [seq x B x 1 x H] - > [seq x B x O x H]

        # set grad for lstm
        grad_wih, grad_whh, grad_bih, grad_bhh = compute_grad_eprop(pEph, ets)

        self.lstm.cell.weight_ih.grad, self.lstm.cell.weight_hh.grad, \
        self.lstm.cell.bias_ih.grad, self.lstm.cell.bias_hh.grad = grad_wih / seq, grad_whh / seq, grad_bih / seq, grad_bhh / seq

        # set grads for linear layer
        self.linear.weight.grad = torch.mean(pEpW, dim=(0, 1))
        return pEph

    def backward_cross_entropy(self, ys, lstm_outputs, caches, labels_seqs, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        tag_scores = F.log_softmax(ys, dim=-1)
        seq, batch_size, _ = lstm_outputs.size()
        idx = (torch.arange(self.output_size)).repeat(seq * batch_size, 1).to(device)  # [seq * B x O]
        labels_repeated = torch.repeat_interleave(labels_seqs[:, None], self.output_size, dim=1)  # [seq * B x O]

        # to generate one-hot encoded vector
        dEdy = -(labels_repeated == idx).float() + torch.exp(
            tag_scores)  # [seq * batch x out] , derivative w.r.t. tag scores
        dy = dEdy.reshape(seq, batch_size, self.output_size)

        dwout = torch.matmul(dy[:, :, :, None], lstm_outputs[:, :, None, :])
        dls = torch.matmul(dy, self.linear.weight.clone())  # [S x B x O] @ [O x H] -> [S x B x H]

        grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh, grad_h = self.lstm.backward(inputs, dls, caches)

        # assign values to gradients
        # use double mean because in pytorch mse--reduction = 'mean
        self.linear.weight.grad = torch.mean(dwout, dim=(0, 1))
        self.lstm.cell.weight_ih.grad = torch.mean(grad_W_ih, dim=(0, 1))
        self.lstm.cell.weight_hh.grad = torch.mean(grad_W_hh, dim=(0, 1))
        self.lstm.cell.bias_ih.grad = torch.mean(grad_b_ih, dim=(0, 1))
        self.lstm.cell.bias_hh.grad = torch.mean(grad_b_hh, dim=(0, 1))
        return grad_h


###############################################################################
# E-prop 1 for char
# Labels are only feed into net at the end of the sequence (t = T)
###############################################################################

class LstmEprop1LastTimestep(Eprop1):

    def __init__(self, LstmLayer, input_size, hidden_size, output_size, batch_first=True,
                 feedback_type='symmetric_feedback'):
        super(LstmEprop1LastTimestep, self).__init__(hidden_size, output_size, feedback_type)
        self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        self.lstm = LstmLayer(LSTMCell, input_size, hidden_size)
        # self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.batch_first = batch_first

    def forward(self, inp, lengths):
        if self.batch_first:
            inp = inp.transpose(0, 1)
        batch_size = inp.size(1)

        state = init_hidden(batch_size, self.hidden_size)
        lstm_outputs, _, ets, _ = self.lstm(inp, state)  # lstm output : [seq x B x H]

        # extract the corresponding values at T
        lstm_output_T = torch.stack([y[l - 1] for (y, l) in zip(lstm_outputs.transpose(0, 1), lengths)])
        ets_wih, ets_whh, ets_bih, ets_bhh = ets
        et_wih_T = torch.stack([y[l - 1] for (y, l) in zip(ets_wih.transpose(0, 1), lengths)])
        et_whh_T = torch.stack([y[l - 1] for (y, l) in zip(ets_whh.transpose(0, 1), lengths)])
        et_bih_T = torch.stack([y[l - 1] for (y, l) in zip(ets_bih.transpose(0, 1), lengths)])
        et_bhh_T = torch.stack([y[l - 1] for (y, l) in zip(ets_bhh.transpose(0, 1), lengths)])

        ys = self.linear(lstm_output_T)  # tag space: [B x Out]
        return ys, lstm_output_T, state, (et_wih_T, et_whh_T, et_bih_T, et_bhh_T)  # for classification

    def forward_eval(self, input, lengths, state):
        if self.batch_first:
            input = input.transpose(0, 1)

        # batch_size = input.size(1)
        # state = init_hidden(batch_size, self.hidden_size)
        lstm_outputs, state = self.lstm.forward_eval(input, state)
        lstm_output_T = torch.stack([y[l - 1] for (y, l) in zip(lstm_outputs.transpose(0, 1), lengths)])
        ys = self.linear(lstm_output_T)  # tag space: [B x Out]
        return ys, state

    def eprop_cross_entropy(self, ys, lstm_output_T, et, labels_T):
        """

        :param ys: [B, O]
        :param lstm_output_T: [B, H ]
        :param et: Tuple
        :param labels_T: [seq * B, 1] only the correct class, not one-hot encoded
        :return:
        """
        tag_scores = F.log_softmax(ys, dim=-1)
        batch_size = tag_scores.size(0)
        idx = (torch.arange(self.output_size)).repeat(batch_size, 1).to(device)  # [B x O]
        labels_repeated = torch.repeat_interleave(labels_T[:, None], self.output_size, dim=1)  # [B x O]

        # to generate one-hot encoded vector
        dEdy = -(labels_repeated == idx).float() + torch.exp(tag_scores)  # [batch x out] , derivative w.r.t. tag scores
        W_linear = getattr(self, self.feedback_type)()

        # compute partial E, h
        pEph = torch.matmul(dEdy, W_linear)  # [B x O] @ [O x H] - > [B x H]
        # partial E, W_linear
        pEpW = torch.matmul(dEdy[:, :, None],
                            lstm_output_T[:, None, :])  # [B x O x 1] @ [B x 1 x H] - > [B x O x H]
        # set grad for lstm
        grad_wih, grad_whh, grad_bih, grad_bhh = compute_grad_eprop_T(pEph, et)

        self.lstm.cell.weight_ih.grad, self.lstm.cell.weight_hh.grad, \
        self.lstm.cell.bias_ih.grad, self.lstm.cell.bias_hh.grad = grad_wih, grad_whh, grad_bih, grad_bhh

        # set grads for linearlayer
        self.linear.weight.grad = torch.mean(pEpW, dim=0)
        # self.linear.bias.grad = torch.mean(pEpb, dim=1)

    def backward_cross_entropy(self, ys, lstm_output_T, caches, labels_T, inputs):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        tag_scores = F.log_softmax(ys, dim=-1)
        batch_size = tag_scores.size(0)
        idx = (torch.arange(self.output_size)).repeat(batch_size, 1).to(device)  # [B x O]
        labels_repeated = torch.repeat_interleave(labels_T[:, None], self.output_size, dim=1)  # [B x O]

        # to generate one-hot encoded vector
        dy = -(labels_repeated == idx).float() + torch.exp(tag_scores)

        dwout = torch.matmul(dy[:, :, None],
                            lstm_output_T[:, None, :])

        dls = torch.matmul(dy, self.linear.weight.clone())  # [S x B x O] @ [O x H] -> [S x B x H]

        grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh, grad_h = self.lstm.backward(inputs, dls, caches)

        # assign values to gradients
        # use double mean because in pytorch mse--reduction = 'mean
        self.linear.weight.grad = torch.mean(dwout, dim=(0, 1))
        self.lstm.cell.weight_ih.grad = torch.mean(grad_W_ih, dim=(0, 1))
        self.lstm.cell.weight_hh.grad = torch.mean(grad_W_hh, dim=(0, 1))
        self.lstm.cell.bias_ih.grad = torch.mean(grad_b_ih, dim=(0, 1))
        self.lstm.cell.bias_hh.grad = torch.mean(grad_b_hh, dim=(0, 1))
        return grad_h


###############################################################################
# RNN with bptt
###############################################################################
class RnnBptt(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_first=False):
        super(RnnBptt, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first

        self.linear_in = torch.nn.Linear(
            input_size, hidden_size, bias=False
        )
        self.linear_hid = torch.nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        self.linear_out = torch.nn.Linear(
            hidden_size, output_size, bias=False
        )

        self.tanh = torch.nn.Tanh()

    def forward(self, x, h):
        if self.batch_first:
            x = x.transpose(0, 1)

        time_steps = x.size(0)
        batch_size = x.size(1)

        output = []
        hidden = []

        for t in range(time_steps):
            x_input = x[t]
            h = self.tanh(
                self.linear_in(x_input) +
                self.linear_hid(h)
            )
            o = self.linear_out(h)
            output.append(o)
            hidden.append(h)

        ys = torch.stack(output).view(-1, self.output_size)
        return ys, h


###############################################################################
# RNN with eprop
###############################################################################


class RnnEprop1(Eprop1):

    def __init__(self, input_size, hidden_size, output_size, batch_first=False, feedback_type='symmetric_feedback'):
        super(RnnEprop1, self).__init__(hidden_size, output_size, feedback_type)

        self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.output_size = output_size
        self.batch_first = batch_first

        self.linear_in = torch.nn.Linear(
            input_size, hidden_size, bias=False
        )
        self.linear_hid = torch.nn.Linear(
            hidden_size, hidden_size, bias=False
        )
        # self.linear = torch.nn.Linear(
        #     hidden_size, output_size, bias=False
        # )

        self.tanh = torch.nn.Tanh()

    def forward(self, x, target, h):
        if self.batch_first:
            x = x.transpose(0, 1)

        time_steps = x.size(0)
        batch_size = x.size(1)

        ev_wih = torch.zeros(batch_size, self.hidden_size, self.input_size, device=device)
        ev_whh = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)

        grad_wih = torch.zeros(batch_size, self.hidden_size, self.input_size, device=device)
        grad_whh = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
        grad_wout = torch.zeros(batch_size, self.output_size, self.hidden_size, device=device)

        W_hh = self.linear_hid.weight.clone()

        output = []
        hidden = []

        W_out = getattr(self, self.feedback_type)()

        for t in range(time_steps):
            x_input = x[t]
            z = target[t]

            D = torch.diag(W_hh) * tanh_derivative(h)  # [B, H]
            ev_wih = D[:, :, None] * ev_wih + x_input[:, None, :]  # [B, H, I]
            ev_whh = D[:, :, None] * ev_whh + h[:, None, :]  # [B, H, H]

            h = self.tanh(
                self.linear_in(x_input) +
                self.linear_hid(h)
            )

            phpnet = tanh_derivative(h)  # [H]
            et_wih = phpnet[:, :, None] * ev_wih  # [B, H, I]
            et_whh = phpnet[:, :, None] * ev_whh  # [B, H, H]

            o = self.linear(h)

            L = torch.matmul((o - z), W_out)  # [B, O] x [B, H] -> [B, H]
            grad_wout += torch.matmul((o - z)[:, :, None], h[:, None, :])  # [B, O, 1] x [B, 1, H]
            grad_wih += L[:, :, None] * et_wih * 1e-3
            grad_whh += L[:, :, None] * et_whh * 1e-5

            output.append(o)
            hidden.append(h)

        self.linear.weight.grad = torch.mean(grad_wout, 0)
        self.linear_in.weight.grad = torch.mean(grad_wih, 0)
        self.linear_hid.weight.grad = torch.mean(grad_whh, 0)

        ys = torch.stack(output).view(-1, self.output_size)  # [seq x out]
        return ys, h

    def forward_eval(self, x, h):
        if self.batch_first:
            x = x.transpose(0, 1)

        time_steps = x.size(0)
        output = []
        hidden = []

        for t in range(time_steps):
            x_input = x[t]

            h = self.tanh(
                self.linear_in(x_input) +
                self.linear_hid(h)
            )
            o = self.linear(h)
            output.append(o)
            hidden.append(h)
        ys = torch.stack(output).view(-1, self.output_size)
        return ys, h
