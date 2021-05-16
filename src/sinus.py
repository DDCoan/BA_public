"""
Sinus regression problem
Data: [1,0...,0], length = 100
Training approaches:
    BPTT
    eprop 1
Network:
    simple RNN
    LSTM
"""
from models import *
import csv
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import os
import argparse
from sys import exit

# torch.manual_seed(1234)
# np.random.seed(1234)
parser = argparse.ArgumentParser(description='Sinus Model')
parser.add_argument('--net', type=str, default='lstm', choices=['lstm', 'rnn'],
                    help='network used to train: lstm or rnn')
parser.add_argument('--model', type=str, default='bptt', choices=['bptt', 'eprop1'],
                    help='learning algorithms that should be used: bptt or eprop1')
parser.add_argument('--feedback_type', type=str, default='symmetric_feedback',
                    choices=['symmetric_feedback', 'random_feedback'],
                    help='type of feedback matrix: symmetric_feedback or random_feedback')
parser.add_argument('--layer_type', type=str, default='LstmC', choices=['LstmC', 'LstmNet'],
                    help='type of e-prop derivation: LstmC or LstmNet')
parser.add_argument('--single_exp', type=str2bool, nargs='?',
                    const=True, default=False,
                    help="activate single experiment mode")
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--log_interval', type=int, default=10,
                    help='length of unit training loop')
parser.add_argument('--runs', type=int, default=10,
                    help='repeat evaluation experiments')
args = parser.parse_args()


# example sinus function
def sin(t):
    return torch.sin(.2 * t)


def plot_sinus(ys, zs):
    plt.plot(zs, 'green', label='target')
    plt.plot(ys, 'red', label='prediction')
    plt.xlabel('t')
    plt.ylabel('sin(t)')
    plt.legend()
    plt.show()


# SET UP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq = 100
inp_train = torch.zeros(seq, dtype=torch.float32, device=device)[:, None, None]  # [seq x B x I ] = [100, 1, 1]
inp_train[0, 0, 0] = 1.
inp_val = torch.zeros(30, device=device)[:, None, None]  # [10, 1, 1]
inp_test = torch.zeros(10, device=device)[:, None, None]  # [10, 1, 1]

target_train = sin(torch.arange(seq, dtype=torch.float32, device=device)[:, None])  # seq x B x I
target_val = sin(torch.arange(100, 130, dtype=torch.float32, device=device)[:, None])
target_test = sin(torch.arange(130, 140, dtype=torch.float32, device=device)[:, None])

inputs = torch.cat((inp_train, inp_val), dim=0)
targets = torch.cat((target_train.view(-1), target_val.view(-1)))

criterion = torch.nn.MSELoss()

current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../')


def lstm_bptt():
    errs = []

    path = '/sinus/lstm/bptt/nhid_{}'.format(args.nhid)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeated_eval.csv'.format(args.nhid), 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr', 'test error'])
        for i in range(args.runs):
            best_err = 10.
            best_hidden = tuple()
            model = LstmBpttSin(args.nhid).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for epoch in range(args.epochs):
                    model.train()
                    for _ in range(args.log_interval):
                        hidden = (
                            torch.zeros((1, 1, model.nhid), device=device),
                            torch.zeros((1, 1, model.nhid), device=device))
                        optimizer.zero_grad()
                        ys_train, hidden = model(inp_train, hidden)
                        loss = criterion(ys_train, target_train[:, :, None])
                        loss.backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        ys_val, hidden = model(inp_val, hidden)
                        loss_val = criterion(ys_val, target_val[:, :, None])
                    # print('epoch {} \t train loss = {:.6f} \t test loss = {:.6f}'.format(epoch + 1, loss.item(),
                    # loss_val.item()))
                    writer.writerow([epoch + 1, loss.item(), loss_val.item()])
                    if loss_val < best_err:
                        best_err = loss_val
                        best_hidden = hidden
                        # ys = torch.cat((ys_train.detach().view(-1), ys_val.view(-1)))
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                ys_test, _ = model(inp_test, best_hidden)
                loss = criterion(ys_test, target_test[:, :, None])
                errs += [loss.item()]
            print('runNr = {} \t test loss = {:.6f}'.format(i, loss.item()))
            writer_eval.writerow([i, loss.item()])

        errs_mean = np.mean(errs)
        errs_std = np.std(errs)
        writer_eval.writerow(['mean', errs_mean.item()])
        writer_eval.writerow(['std', errs_std.item()])
        # plot_sinus(ys, targets)
    return errs, errs_mean, errs_std


def single_exp_lstm_eprop1():
    path = '/sinus/lstm/single_eprop/nhid_{}'.format(args.nhid)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    model = LstmEprop1AllTimesteps(globals()[args.layer_type], input_size=1, hidden_size=args.nhid, output_size=1,
                                   batch_first=False, feedback_type=args.feedback_type).to(device)

    ################# weights * 1/spectral radius ##############
    # whh = model.lstm.cell.weight_hh.clone()
    # u, s, vh = np.linalg.svd(whh.detach().numpy())
    # print('spectral radius = {:.2f}'.format(s[0]**2))
    # model.lstm.cell.weight_hh = torch.nn.Parameter(data = whh / (s[0]**2))
    #############################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)

    best_err = 10.

    # switch-off backprop
    def enable_autograd(is_using_bptt):
        for param in model.parameters():
            param.requires_grad = is_using_bptt

    # initialization, only True when need information from bptt, otherwise False for eprop
    enable_autograd(True)

    checkpoints = [int(1 / 3 * args.epochs), int(2 / 3 * args.epochs), args.epochs - 1]
    ets_checkpoints = {
        'weight_ih': [],
        'weight_hh': [],
        'bias_ih': [],
        'bias_hh': []
    }
    eprop_grad_checkpoints = {
        'weight_ih': [],
        'weight_hh': [],
        'bias_ih': [],
        'bias_hh': []
    }  # only for parameters : weight_ih, hh,  bias_ih, hh

    bptt_grad_checkpoints = {
        'weight_ih': [],
        'weight_hh': [],
        'bias_ih': [],
        'bias_hh': []
    }
    eprop_ls_checkpoints = []
    bptt_ls_checkpoints = []

    def save_ets_checkpoints(ets):
        ets_checkpoints['weight_ih'] += [ets[0]]
        ets_checkpoints['weight_hh'] += [ets[1]]
        ets_checkpoints['bias_ih'] += [ets[2]]
        ets_checkpoints['bias_hh'] += [ets[3]]

    def save_grad_checkpoints(dicts, Ls, ets):
        ets_wih, ets_whh, ets_bih, ets_bhh = ets
        # et_wih size is 4*hidden so repeat Ls 4 times in last dim
        Ls = Ls.repeat(1, 1, 4)

        dicts['weight_ih'] += [ets_wih * Ls[:, :, :, None]]
        dicts['weight_hh'] += [ets_whh * Ls[:, :, :, None]]
        dicts['bias_ih'] += [ets_bih * Ls]
        dicts['bias_hh'] += [ets_bih * Ls]

    with open(res_path + '/res.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for epoch in range(args.epochs):
            for _ in range(args.log_interval):  # set log_intervals = 1 for tracing et and gradients
                optimizer.zero_grad()
                seq, batch_size, _ = inp_train.size()
                ys_train, lstm_outputs, state, ets, caches = model(inp_train)
                # torch.save(ys_train - target_train, current_dir + '/y_diff.pt')
                # exit(0)
                loss_train = criterion(ys_train[:, :, None], target_train[:, :, None])
                ##############################################
                """bptt_start_configuration"""
                # if i < 100:
                #     loss_train.backward()
                # else:
                #     model.eprop_mse(ys_train, lstm_outputs, ets, target_train)
                ##############################################
                # model.eprop_mse(ys_train, lstm_outputs, ets, target_train)
                # optimizer.step()
                ##############################################
                """store checkpoints --training with eprop"""
                if epoch in checkpoints:
                    Ls_eprop = model.eprop_mse(ys_train, lstm_outputs, ets, target_train)
                    eprop_ls_checkpoints += [Ls_eprop]
                    save_ets_checkpoints(ets)
                    save_grad_checkpoints(eprop_grad_checkpoints, Ls_eprop, ets)
                    # rerun backprop
                    optimizer.zero_grad()
                    Ls_bptt = model.backward_mse(ys_train, lstm_outputs, caches, target_train, inp_train)
                    bptt_ls_checkpoints += [Ls_bptt]
                    save_grad_checkpoints(bptt_grad_checkpoints, Ls_bptt, ets)
                    optimizer.zero_grad()
                ##############################################
                # model.backward_mse(ys_train, lstm_outputs, caches, target_train, inp_train)
                loss_train.backward()
                optimizer.step()
                ##############################################
            ############# check spectral radius ###############
            # whh = model.lstm.cell.weight_hh.clone().detach().numpy()
            # u,s,vh = np.linalg.eig(whh)
            # print('spectral radius = {:.2f}' .format(s[0]**2))
            ###################################################

            # obs: wih grad for forget gate = 0, both for bptt and eprop
            # problem: last two elements of grad wout too large ~ 10
            model.eval()
            # print(state) # observation: h is normal, c very large
            with torch.no_grad():
                ys_val, state = model.forward_eval(inp_val, state)
                loss_val = criterion(ys_val[:, :, None], target_val[:, :, None])
            # decrease then increase then decrease
            print('epoch {} \t train loss = {:.6f} \t test loss = {:.6f}'.format(epoch + 1, loss_train.item(),
                                                                                 loss_val.item()))
            writer.writerow([epoch + 1, loss_train.item(), loss_val.item()])
            if loss_val < best_err:
                best_err = loss_val
                ys = torch.cat((ys_train.detach().view(-1), ys_val.view(-1)))
                torch.save(model.state_dict(), model_path + '/model.pt')
    plot_sinus(ys.detach().numpy(), targets)
    print('best error of lstm eprop ' + str(best_err))

    checkpints_path = exp_path + '/checkpoints'
    create_folder_when_necessary(checkpints_path)
    torch.save(ets_checkpoints, checkpints_path + '/ets_checkpoints.pt')
    torch.save(eprop_grad_checkpoints, checkpints_path + '/eprop_grad_checkpoints.pt')
    torch.save(bptt_grad_checkpoints, checkpints_path + '/bptt_grad_checkpoints.pt')
    torch.save(eprop_ls_checkpoints, checkpints_path + '/eprop_ls_checkpoints.pt')
    torch.save(bptt_ls_checkpoints, checkpints_path + '/bptt_ls_checkpoints.pt')
    return best_err


def lstm_eprop1():
    def init_model():
        mdl = LstmEprop1AllTimesteps(globals()[args.layer_type], input_size=1, hidden_size=args.nhid, output_size=1,
                                     batch_first=False, feedback_type=args.feedback_type).to(device)
        return mdl

    errs = []

    path = '/sinus/lstm/eprop1/{}/{}/nhid_{}'.format(args.feedback_type, args.layer_type, args.nhid)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeated_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr', 'test error'])
        for i in range(args.runs):
            best_err = 10.
            best_state = tuple()
            model = init_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # disable backprop
            for param in model.parameters():
                param.requires_grad = False
            # for param in model.parameters():
            #     param.requires_grad = (i < int(.1 * args.epochs))
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for epoch in range(args.epochs):
                    for _ in range(args.log_interval):  # set log_intervals = 1 for tracing et and gradients
                        optimizer.zero_grad()
                        _, batch_size, _ = inp_train.size()
                        ys_train, lstm_outputs, state, ets, caches = model(inp_train)
                        # lstm_outputs.retain_grad()
                        loss_train = criterion(ys_train[:, :, None], target_train[:, :, None])
                        # if i < int(.1 * args.epochs):
                        #     loss_train.backward()
                        # else:
                        #     model.eprop_mse(ys_train, lstm_outputs, ets, target_train)
                        model.eprop_mse(ys_train, lstm_outputs, ets, target_train)
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        ys_val, state = model.forward_eval(inp_val, state)
                        loss_val = criterion(ys_val[:, :, None], target_val[:, :, None])
                    # decrease then increase then decrease
                    print('epoch {} \t train loss = {:.6f} \t test loss = {:.6f}'.format(epoch + 1, loss_train.item(),
                                                                                         loss_val.item()))
                    writer.writerow([epoch + 1, loss_train.item(), loss_val.item()])
                    # try: if use train error as criterion
                    # if loss_train < best_err:
                    #     best_err = loss_train
                    if loss_val < best_err:
                        best_err = loss_val
                        best_state = state
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                ys, _ = model.forward_eval(inp_test, best_state)
                loss = criterion(ys[:, :, None], target_test[:, :, None])
                errs += [loss.item()]
            print('runNr = {} \t best validation loss = {:.6f} \t test loss = {:.6f}'.format(i, best_err.item(),
                                                                                             loss.item()))
            writer_eval.writerow([i, loss.item()])

        errs_mean = np.mean(errs)
        errs_std = np.std(errs)
        writer_eval.writerow(['mean', errs_mean.item()])
        writer_eval.writerow(['std', errs_std.item()])
        # plot_sinus(ys, targets)
    return errs, errs_mean, errs_std


def rnn_bptt():
    errs = []

    path = '/sinus/rnn/bptt/nhid_{}'.format(args.nhid)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeated_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr', 'test error'])
        for i in range(args.runs):
            best_err = 10.
            # best_hidden = tuple()
            model = RnnBptt(1, args.nhid, 1, batch_first=False).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for epoch in range(args.epochs):
                    model.train()
                    for _ in range(args.log_interval):
                        optimizer.zero_grad()
                        _, batch_size, _ = inp_train.size()
                        h = torch.zeros(batch_size, model.hidden_size, device=device)
                        ys_train, h = model(inp_train, h)
                        loss_train = criterion(ys_train, target_train)
                        loss_train.backward()
                        optimizer.step()
                    # validation
                    model.eval()
                    ys_val, h = model(inp_val, h)
                    loss_val = criterion(ys_val, target_val)
                    print('epoch {} \t train loss = {:.6f} \t test loss = {:.6f}'.format(epoch + 1, loss_train.item(),
                                                                                         loss_val.item()))
                    writer.writerow([epoch + 1, loss_train.item(), loss_val.item()])
                    if loss_val < best_err:
                        best_err = loss_val
                        best_hidden = h
                        # ys = torch.cat((ys_train.view(-1), ys_val.view(-1)))
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                ys, _ = model(inp_test, best_hidden)
                loss = criterion(ys[:, :, None], target_test[:, :, None])
                errs += [loss.item()]
            print('runNr = {} \t test loss = {:.6f}'.format(i, loss.item()))
            writer_eval.writerow([i, loss.item()])

        errs_mean = np.mean(errs)
        errs_std = np.std(errs)
        writer_eval.writerow(['mean', errs_mean.item()])
        writer_eval.writerow(['std', errs_std.item()])
        # plot_sinus(ys, targets)

    # print('test with forward_eval,-3,-5')
    # print('best error of rnn ' + str(best_err))
    return errs, errs_mean, errs_std


def rnn_eprop1():
    errs = []
    path = '/sinus/rnn/eprop1/{}/nhid_{}'.format(args.feedback_type, args.nhid)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeated_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr', 'test error'])
        for i in range(args.runs):
            best_err = 10.
            # best_hidden = tuple()
            model = RnnEprop1(1, args.nhid, 1, batch_first=False, feedback_type=args.feedback_type).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # disable backprop
            for param in model.parameters():
                param.requires_grad = False

            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                for epoch in range(args.epochs):
                    model.train()
                    for _ in range(args.log_interval):
                        optimizer.zero_grad()
                        _, batch_size, _ = inp_train.size()
                        h = torch.zeros(batch_size, model.hidden_size, device=device)
                        ys_train, h = model(inp_train, target_train, h)
                        loss_train = criterion(ys_train, target_train)
                        optimizer.step()
                    # validation
                    model.eval()
                    ys_val, h = model.forward_eval(inp_val, h)
                    loss_val = criterion(ys_val, target_val)
                    print('epoch {} \t train loss = {:.6f} \t test loss = {:.6f}'.format(epoch + 1, loss_train.item(),
                                                                                         loss_val.item()))
                    writer.writerow([epoch + 1, loss_train.item(), loss_val.item()])
                    if loss_val < best_err:
                        best_err = loss_val
                        best_hidden = h
                        # ys = torch.cat((ys_train.view(-1), ys_val.view(-1)))
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                ys, _ = model.forward_eval(inp_test, best_hidden)
                loss = criterion(ys[:, :, None], target_test[:, :, None])
                errs += [loss.item()]
            print('runNr = {} \t test loss = {:.6f}'.format(i, loss.item()))
            writer_eval.writerow([i, loss.item()])

        errs_mean = np.mean(errs)
        errs_std = np.std(errs)
        writer_eval.writerow(['mean', errs_mean.item()])
        writer_eval.writerow(['std', errs_std.item()])
        # plot_sinus(ys, targets)

    # print('test with forward_eval,-3,-5')
    # print('best error of rnn ' + str(best_err))
    return errs, errs_mean, errs_std


#
# torch.save(ets_checkpoints, current_dir + '/ets_checkpoints.pt')
# torch.save(eprop_grad_checkpoints, current_dir + '/eprop_grad_checkpoints.pt')
# torch.save(bptt_grad_checkpoints, current_dir + '/bptt_grad_checkpoints.pt')
# torch.save(eprop_ls_checkpoints, current_dir + '/eprop_ls_checkpoints.pt')
# torch.save(bptt_ls_checkpoints, current_dir + '/bptt_ls_checkpoints.pt')
# torch.save(ys, current_dir + '/sinus_prediction.pt')

current_dir += '/test'

if __name__ == "__main__":
    if args.single_exp:
        func = globals()['single_exp_lstm_eprop1']
    else:
        func = globals()[args.net + '_' + args.model]
    func()
