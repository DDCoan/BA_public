"""
Basic calculations related to eprop1
"""

from utils import *
from torch import Tensor
from typing import Tuple
"""DERIVATION IN TERMS OF  C """

"""
computation of eligibility vector 
"""


def compute_ev_c(evx: Tuple[Tensor], gates: Tuple[Tensor], hx: Tensor, cx: Tensor, inp: Tensor):
    ingate, forgetgate, cellgate, outgate = gates

    evx_weight_ii, evx_weight_if, evx_weight_ic, evx_weight_hi, evx_weight_hf, evx_weight_hc, \
    evx_bias_ii, evx_bias_if, evx_bias_ic, evx_bias_hi, evx_bias_hf, evx_bias_hc = evx

    # compute  evy_weight_ii and evy_bias_ii
    pcpi = cellgate * sigmoid_derivative(ingate)  # partial c / partial ingate
    evy_weight_ii = evx_weight_ii * forgetgate[:, :, None] + torch.matmul(pcpi[:, :, None], inp[:, None, :])
    evy_bias_ii = evx_bias_ii * forgetgate + pcpi

    # compute evy_weight_hi and evy_bias_hi
    evy_weight_hi = evx_weight_hi * forgetgate[:, :, None] + torch.matmul(pcpi[:, :, None], hx[:, None, :])
    evy_bias_hi = evx_bias_hi * forgetgate + pcpi

    # compute  evy_weight_if and evy_bias_if
    pcpf = cx * sigmoid_derivative(forgetgate)
    evy_weight_if = evx_weight_if * forgetgate[:, :, None] + torch.matmul(pcpf[:, :, None], inp[:, None, :])
    evy_bias_if = evx_bias_if * forgetgate + pcpf

    # compute  evy_weight_hf and evy_bias_hf
    evy_weight_hf = evx_weight_hf * forgetgate[:, :, None] + torch.matmul(pcpf[:, :, None], hx[:, None, :])
    evy_bias_hf = evx_bias_hf * forgetgate + pcpf

    # compute evy_weight_ig, evy_bias_ig
    pcpg = ingate * tanh_derivative(cellgate)
    evy_weight_ic = evx_weight_ic * forgetgate[:, :, None] + torch.matmul(pcpg[:, :, None], inp[:, None, :])
    evy_bias_ic = evx_bias_ic * forgetgate + pcpg

    # compute evy_weight_hc, evy_bias_hc
    evy_weight_hc = evx_weight_hc * forgetgate[:, :, None] + torch.matmul(pcpg[:, :, None], hx[:, None, :])
    evy_bias_hc = evx_bias_hc * forgetgate + pcpg

    return evy_weight_ii, evy_weight_if, evy_weight_ic, evy_weight_hi, evy_weight_hf, evy_weight_hc, \
           evy_bias_ii, evy_bias_if, evy_bias_ic, evy_bias_hi, evy_bias_hf, evy_bias_hc


"""
computation of eligibility traces based on derivation in terms of c
"""


def compute_et_c(evy: Tensor, gates: Tuple[Tensor], hx: Tensor, cy, inp: Tensor):
    ingate, forgetgate, cellgate, outgate = gates

    evy_weight_ii, evy_weight_if, evy_weight_ic, evy_weight_hi, evy_weight_hf, evy_weight_hc, \
    evy_bias_ii, evy_bias_if, evy_bias_ic, evy_bias_hi, evy_bias_hf, evy_bias_hc = evy

    # compute eligibility traces for in-, cell- and forget gate
    phpc = outgate * tanh_derivative(
        torch.tanh(cellgate))  # partial derivative of current lstm out w.r.t. current hidden state

    phpc3 = phpc.repeat(1, 3)
    et_weight_iifc = torch.cat([evy_weight_ii, evy_weight_if, evy_weight_ic], dim=1) * phpc3[:, :, None]
    et_bias_iifc = torch.cat([evy_bias_ii, evy_bias_if, evy_bias_ic], dim=1) * phpc3

    et_weight_hifc = torch.cat([evy_weight_hi, evy_weight_hf, evy_weight_hc], dim=1) * phpc3[:, :, None]
    et_bias_hifc = torch.cat([evy_bias_hi, evy_bias_hf, evy_bias_hc], dim=1) * phpc3

    # compute for io, ho
    phpo = torch.tanh(cy) * sigmoid_derivative(outgate)
    # print('phpo')
    # print(phpo)
    # phpo : non-zeros, but inp[:,:,3] = 0
    et_weight_io = torch.matmul(phpo[:, :, None], inp[:, None, :])
    et_bias_io = phpo.clone()
    et_weight_ho = torch.matmul(phpo[:, :, None], hx[:, None, :])
    et_bias_ho = phpo.clone()

    # concatenate results
    et_weight_ih = torch.cat([et_weight_iifc, et_weight_io], 1)
    et_weight_hh = torch.cat([et_weight_hifc, et_weight_ho], 1)
    et_bias_ih = torch.cat([et_bias_iifc, et_bias_io], 1)
    et_bias_hh = torch.cat([et_bias_hifc, et_bias_ho], 1)

    return et_weight_ih, et_weight_hh, et_bias_ih, et_bias_hh


"""DERIVATION IN TERMS OF  NET """


def compute_ph_pnet(c, cx, gates):
    """
    compute the derivative of current h w.r.t. current net activation
    :param c: hidden state [B, H]
    :param cx: last hidden state [B, H]
    :param gates: current gates [B, 4H]
    :return: partial derivatives [B, 3H]
    """

    ingate, forgetgate, cellgate, outgate = gates
    pc_pgate = torch.cat([cellgate, cx, ingate], dim=1)  # [B x H]

    pgate_pnet = torch.cat([sigmoid_derivative(ingate),
                            sigmoid_derivative(forgetgate),
                            tanh_derivative(cellgate)], dim=1)
    return outgate.repeat(1, 3) * (tanh_derivative(torch.tanh(c))).repeat(1, 3) * pc_pgate * pgate_pnet


def compute_ev_net(evx, W_hh, gates_x, hx, cx, cx_, x):
    """
    compute eligibility vector of parameters
    :param evx: ev_{t - 1}
    :param W_hh: hidden weights
    :param gates_x: last gates ~ {t - 1}
    :param hx: last lstm output ~ h_{t - 1}
    :param cx: last hidden state ~ c_{t - 1}
    :param cx_: c_{t - 2}
    :param x: current net input ~ x_t
    :return: concatenated eligibility vectors of input-, forget- and cellgate
    """
    ev_wiifg, ev_whifg, ev_biifg, ev_bhifg = evx  # [4H x I], [4H x H], 4H, 4H

    pnet_phx = torch.diag(W_hh)

    phx_pnetx = compute_ph_pnet(cx, cx_, gates_x)
    Dx = pnet_phx.repeat(1, 3) * phx_pnetx

    ev_wiifg = Dx[:, :, None] * ev_wiifg + x[:, None, :]
    ev_whifg + Dx[:, :, None] * ev_whifg + hx[:, None, :]
    ev_biifg = Dx * ev_biifg + 1
    ev_bhifg = Dx * ev_bhifg + 1

    return ev_wiifg, ev_whifg, ev_biifg, ev_bhifg


def compute_et_net(ev, gates, c, hx, cx, x):
    """
    compute eligibility traces of parameters
    :param ev: ev_t for i,f,g
    :param gates: current gates information
    :param c: c_t
    :param hx: h_{t - 1}
    :param cx: c_{t - 1}
    :param x: x_t
    :return: et for all weights and bias
    """
    ev_wiifg, ev_whifg, ev_biifg, ev_bhifg = ev
    ingate, forgetgate, cellgate, outgate = gates

    # compute for input-, forget- and cell gate
    ph_pnet = compute_ph_pnet(c, cx, gates)

    et_wiifg = ph_pnet[:, :, None] * ev_wiifg
    et_whifg = ph_pnet[:, :, None] * ev_whifg
    et_biifg = ph_pnet * ev_biifg
    et_bhifg = ph_pnet * ev_bhifg

    # compute for output gate
    phpo = torch.tanh(c) * sigmoid_derivative(outgate)  # [B, H]
    et_wio = torch.matmul(phpo[:, :, None], x[:, None, :])
    et_who = torch.matmul(phpo[:, :, None], hx[:, None, :])

    # concatenate the et over gates
    et_wih = torch.cat([et_wiifg, et_wio], dim=1)
    et_whh = torch.cat([et_whifg, et_who], dim=1)
    et_bih = torch.cat([et_biifg, phpo], dim=1)
    et_bhh = torch.cat([et_bhifg, phpo], dim=1)
    return et_wih, et_whh, et_bih, et_bhh


"""compute error signals"""


def compute_grad_eprop(Ls, ets):
    """Returns approximate gradients of e-prop for all sequenced labels from 1 to T"""
    """Args: 
        Lr: learning signals [seq x B x H]
        ets: eigibility_traces 
             weight_ih : [seq x B x 4*H x I]
             weight_hh : [seq x B x 4*H x H]
             bias_ih.  : [seq x B x 4*H]
             bias_hh.  : [seq x B x 4*H]
    """
    ets_wih, ets_whh, ets_bih, ets_bhh = ets
    # et_wih size is 4*hidden so repeat Ls 4 times in last dim
    Ls = Ls.repeat(1, 1, 4)

    # sum over t
    grad_wih = torch.sum(torch.mean(ets_wih * Ls[:, :, :, None], dim=1), dim=0)
    grad_whh = torch.sum(torch.mean(ets_whh * Ls[:, :, :, None], dim=1), dim=0)
    grad_bih = torch.sum(torch.mean(ets_bih * Ls, dim=1), dim=0)
    grad_bhh = torch.sum(torch.mean(ets_bhh * Ls, dim=1), dim=0)
    return grad_wih, grad_whh, grad_bih, grad_bhh


def compute_grad_eprop_T(L, et):
    """Returns approximate gradients of e-prop"""
    """Args: 
        Lr: learning signals [B x H] at T
        ets: eigibility_traces 
             weight_ih : [seq x B x 4*H x I]
             weight_hh : [seq x B x 4*H x H]
             bias_ih.  : [seq x B x 4*H]
             bias_hh.  : [seq x B x 4*H]
    """
    # eligibility traces from last time step T
    et_wih_T, et_whh_T, et_bih_T, et_bhh_T = et  # [B x 4H x I]

    # et_wih size is 4*hidden so repeat Ls 4 times in last dim
    L = L.repeat(1, 4)

    # sum over t
    grad_wih = torch.mean(et_wih_T * L[:, :, None], dim=0)
    grad_whh = torch.mean(et_whh_T * L[:, :, None], dim=0)
    grad_bih = torch.mean(et_bih_T * L, dim=0)
    grad_bhh = torch.mean(et_bhh_T * L, dim=0)
    return grad_wih, grad_whh, grad_bih, grad_bhh
