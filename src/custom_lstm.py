"""
Customized LSTM framework, specialised for eprop1
Two types of  LSTM layers:
    LSTM + C derivation
    LSTM + Net derivation
"""

import torch.jit as jit
from utils_eprop import *
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        bias_ih = torch.randn(4 * hidden_size, 1)
        bias_hh = torch.randn(4 * hidden_size, 1)

        torch.nn.init.xavier_normal_(self.weight_ih)
        torch.nn.init.xavier_normal_(self.weight_hh)
        torch.nn.init.xavier_normal_(bias_ih)
        torch.nn.init.xavier_normal_(bias_hh)

        # set bias for foget gate = 1
        # bias_ih[hidden_size : 2 * hidden_size, 0] = torch.ones(hidden_size)
        # bias_hh[hidden_size: 2 * hidden_size, 0] = torch.ones(hidden_size)

        self.bias_ih = torch.nn.Parameter(bias_ih[:, 0])
        self.bias_hh = torch.nn.Parameter(bias_hh[:, 0])

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]):
        hx, cx = state

        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)  # dim : B x I @ I x 4H -> B x 4H

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy), (ingate, forgetgate, cellgate, outgate)

    def backward(self, dl, Dh, dcy, cache, gatesy, statex, inp, dczdcy=0):
        """
        :param inp: input
        :param dczdcy: total derivative of c_{t + 2} w.r.t. c_{t + 1}
        :param statex: (h_{t - 1}, c_{t - 1})
        :param dl: grad from linear layer
        :param Dh: grad from future time steps
        :param dcy: grad of hidden state in next time step
        :param cache: output of forward pass
        :param gatesy: gates of t+1 time step
        :return: current grads
        """
        hx, cx = statex
        state, gates = cache
        h, c = state
        ingate, forgetgate, cellgate, outgate = gates
        # (h, c), (ingate, forgetgate, cellgate, outgate) = cache[0], cache[1]
        iny, fy, gy, oy = gatesy
        tanhc = torch.tanh(c)

        dh = dl + Dh
        dc = dh * outgate * tanh_derivative(tanhc) + dcy * fy  # cellgate ~ g
        # print('hi')
        # print(dh, outgate, tanh_derivative(tanhc), dcy, fy)
        # dc = dh * outgate * tanh_derivative(tanhc) + dcy * forgetgate
        dg = dc * ingate * tanh_derivative(cellgate)
        di = dc * cellgate * sigmoid_derivative(ingate)
        df = dc * cx * sigmoid_derivative(forgetgate)
        do = dh * torch.tanh(c) * sigmoid_derivative(outgate)

        dcx = dc * forgetgate

        dgates = torch.cat([di, df, dg, do], dim=1)  # [B x 4H]

        dinp = torch.matmul(dgates, self.weight_ih)  # [4H x I] x [4H x H]
        Dh = torch.matmul(dgates, self.weight_hh)  # [B x 4H] x [4H x H]

        dW_ih = torch.matmul(dgates[:, :, None], inp[:, None, :])  # [B x 4H x I]
        dW_hh = torch.matmul(dgates[:, :, None], hx[:, None, :])  # [B x 4H x H]

        """useful code, but not for normal BPTT"""
        # compute dct+1/dh
        # W_hi, W_hf, W_hg, _ = self.weight_hh.chunk(4, dim=0)
        # pcypc = fy
        # dcydh = torch.mean(gy * sigmoid_derivative(iny), 0)[:, None] * W_hi \
        #         + torch.mean(c * sigmoid_derivative(fy), 0)[:, None] * W_hf \
        #         + torch.mean(iny * tanh_derivative(gy), 0)[:, None] * W_hg  # [H x H]
        #
        # dhdc = torch.diag(torch.mean(tanh_derivative(tanhc) * outgate, 0))
        #
        # dczdh = torch.matmul(dczdcy, dcydh)  # [H x H] x [H x H]  -> [H x H]
        #
        # dczdc = torch.diag(torch.mean(pcypc, 0)) + torch.matmul(dcydh, dhdc)  # [H x H] + [H x H]

        # return dh, Dh, dc, dinp, dW_ih, dW_hh, dgates, dczdh, dczdc

        return dW_ih, dW_hh, dgates, dh, Dh, dc


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, *args):
        pass

    def forward_eval(self, input, state):
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            inp = inputs[i]
            out, state, _ = self.cell(inp, state)
            outputs += [out]
        return torch.stack(outputs, dim=0), state


"""DERIVATION BASED ON C"""


class LstmC(LSTMLayer):
    def __init__(self, cell, *cell_args):
        super(LstmC, self).__init__(cell, *cell_args)

    def _init_ev(self, batch_size):
        hidden_size = self.cell.hidden_size
        input_size = self.cell.input_size
        # init eligibility vector, order : (W_ii, W_if|W_ic| W_io) because ii and if use the same functions
        ev_weight_ii = torch.zeros(batch_size, hidden_size, input_size, device=device)
        ev_weight_if = torch.zeros(batch_size, hidden_size, input_size, device=device)
        ev_weight_ic = torch.zeros(batch_size, hidden_size, input_size, device=device)

        ev_weight_hi = torch.zeros(batch_size, hidden_size, hidden_size, device=device)
        ev_weight_hf = torch.zeros(batch_size, hidden_size, hidden_size, device=device)
        ev_weight_hc = torch.zeros(batch_size, hidden_size, hidden_size, device=device)

        ev_bias_ii = torch.zeros(batch_size, hidden_size, device=device)
        ev_bias_if = torch.zeros(batch_size, hidden_size, device=device)
        ev_bias_ic = torch.zeros(batch_size, hidden_size, device=device)

        ev_bias_hi = torch.zeros(batch_size, hidden_size, device=device)
        ev_bias_hf = torch.zeros(batch_size, hidden_size, device=device)
        ev_bias_hc = torch.zeros(batch_size, hidden_size, device=device)
        return ev_weight_ii, ev_weight_if, ev_weight_ic, ev_weight_hi, ev_weight_hf, ev_weight_hc, \
               ev_bias_ii, ev_bias_if, ev_bias_ic, ev_bias_hi, ev_bias_hf, ev_bias_hc

    def forward(self, input, state):
        inputs = input.unbind(0)
        seq, batch_size, _ = input.size()
        outputs = torch.jit.annotate(List[Tensor], [])
        caches = torch.jit.annotate(List[Tensor], [])
        ets_wih, ets_whh, ets_bih, ets_bhh = torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], [])

        ev = self._init_ev(batch_size)

        for i in range(len(inputs)):
            hx, cx = state
            inp = inputs[i]
            out, state, gates = self.cell(inp, state)
            cy = state[1]
            outputs += [out]
            caches += [(state, gates)]

            fut_ev = torch.jit._fork(compute_ev_c, ev, gates, hx, cx, inp)
            ev = torch.jit._wait(fut_ev)
            fut_et = torch.jit._fork(compute_et_c, ev, gates, hx, cy, inp)
            et = torch.jit._wait(fut_et)
            et_weight_ih, et_weight_hh, et_bias_ih, et_bias_hh = et
            ets_wih += [et_weight_ih]
            ets_whh += [et_weight_hh]
            ets_bih += [et_bias_ih]
            ets_bhh += [et_bias_hh]
        ets = (torch.stack(ets_wih, dim=0), torch.stack(ets_whh, dim=0), torch.stack(ets_bih, dim=0),
               torch.stack(ets_bhh, dim=0))
        caches += [((torch.zeros(batch_size, self.cell.hidden_size, device=device),
                     torch.zeros(batch_size, self.cell.hidden_size, device=device)),
                    (torch.zeros(batch_size, self.cell.hidden_size, device=device),
                     torch.zeros(batch_size, self.cell.hidden_size, device=device),
                     torch.zeros(batch_size, self.cell.hidden_size, device=device),
                     torch.zeros(batch_size, self.cell.hidden_size, device=device)))]
        return torch.stack(outputs, dim=0), state, ets, caches

    def backward(self, input, dls, caches):
        inputs = input.unbind(0)
        seq, batch_size, _ = input.size()

        hidden_size = self.cell.hidden_size

        # initialize grad tensors
        Dh = torch.zeros(batch_size, hidden_size, device=device)
        dc = torch.zeros(batch_size, hidden_size, device=device)
        grad_h = torch.zeros(seq, batch_size, hidden_size, device=device)
        # grad_c = torch.zeros(seq, batch_size, hidden_size)

        grad_W_ih = torch.zeros(seq, batch_size, 4 * hidden_size, self.cell.input_size, device=device)
        grad_W_hh = torch.zeros(seq, batch_size, 4 * hidden_size, hidden_size, device=device)
        grad_b_ih = torch.zeros(seq, batch_size, 4 * hidden_size, device=device)
        # grad_inp = torch.zeros(seq, batch_size, self.cell.input_size, device=device)
        # grad_ch = torch.zeros(seq, batch_size, hidden_size, hidden_size, device=device)

        for i in range(seq - 1, -1, -1):
            inp = inputs[i]
            gatesy = caches[i + 1][1]
            cache = caches[i]
            statex = caches[i - 1][0]  # i = 0, cache[-1] is filled zeros, fulfilled the expectation
            dl = dls[i]
            dW_ih, dW_hh, dgates, dh, Dh, dc = self.cell.backward(dl, Dh, dc, cache, gatesy, statex, inp)
            # store grads into gradient matrix
            grad_h[i] = dh
            grad_W_ih[i] = dW_ih
            grad_W_hh[i] = dW_hh
            grad_b_ih[i] = dgates
            # grad_c[i] = dc
            # grad_inp[i] = dinp
            # grad_ch[i] = dczdh
        grad_b_hh = grad_b_ih.clone()
        return grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh, grad_h


"""DERIVATION BASED ON NET """


class LstmNet(LSTMLayer):
    def __init__(self, cell, *cell_args):
        super(LstmNet, self).__init__(cell, *cell_args)

    def _init_gates(self, batch_size):
        hidden_size = self.cell.hidden_size
        ingate, forgetgate, cellgate, outgate = torch.zeros(batch_size, 4 * hidden_size, device=device).chunk(4, dim=1)
        return ingate, forgetgate, cellgate, outgate

    def _init_ev(self, batch_size):
        hidden_size = self.cell.hidden_size
        input_size = self.cell.input_size
        # init eligibility vector, order : (W_ii, W_if|W_ic| W_io)
        ev_wiifg = torch.zeros(batch_size, 3 * hidden_size, input_size, device=device)
        ev_whifg = torch.zeros(batch_size, 3 * hidden_size, hidden_size, device=device)
        ev_biifg = torch.zeros(3 * hidden_size, device=device)
        ev_bhifg = torch.zeros(3 * hidden_size, device=device)
        return ev_wiifg, ev_whifg, ev_biifg, ev_bhifg

    def forward(self, input, state):
        inputs = input.unbind(0)
        seq, batch_size, _ = input.size()
        outputs = torch.jit.annotate(List[Tensor], [])
        ets_wih, ets_whh, ets_bih, ets_bhh = torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], []), \
                                             torch.jit.annotate(List[Tensor], [])

        ev = self._init_ev(batch_size)
        W_hh = self.cell.weight_hh
        gates_x = self._init_gates(batch_size)
        cx_ = torch.zeros(batch_size, self.cell.hidden_size, device=device)

        for i in range(len(inputs)):
            hx, cx = state
            inp = inputs[i]
            out, state, gates = self.cell(inp, state)
            c = state[1]
            outputs += [out]

            fut_ev = torch.jit._fork(compute_ev_net, ev, W_hh, gates_x, hx, cx, cx_, inp)
            ev = torch.jit._wait(fut_ev)
            fut_et = torch.jit._fork(compute_et_net, ev, gates, c, hx, cx, inp)
            et = torch.jit._wait(fut_et)
            et_weight_ih, et_weight_hh, et_bias_ih, et_bias_hh = et

            # update parameters
            cx_ = cx
            gates_x = gates

            ets_wih += [et_weight_ih]
            ets_whh += [et_weight_hh]
            ets_bih += [et_bias_ih]
            ets_bhh += [et_bias_hh]
        ets = (torch.stack(ets_wih, dim=0), torch.stack(ets_whh, dim=0), torch.stack(ets_bih, dim=0),
               torch.stack(ets_bhh, dim=0))
        return torch.stack(outputs, dim=0), state, ets, 0  # zero: to make sure the number of output stay consistent
