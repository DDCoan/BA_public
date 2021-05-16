"""
Character sequence classification problem
Data: http://archive.ics.uci.edu/ml/datasets/Character+Trajectories
Training approaches:
    BPTT
    eprop 1
Network:
    LSTM
"""
import time
import os
import numpy as np
import argparse
import csv
from torch.utils import data
from models import *
from custom_dataset import *
from custom_lstm import *
from sys import exit

N = 2858

parser = argparse.ArgumentParser(description='LSTM Classifier Model')
parser.add_argument('--model', type=str, default='bptt', choices=['bptt', 'eprop1'],
                    help='learning algorithms that should be used: bptt or eprop1')
parser.add_argument('--feedback_type', type=str, default='symmetric_feedback',
                    choices=['symmetric_feedback', 'random_feedback'],
                    help='type of feedback matrix: symmetric_feedback or random_feedback')
parser.add_argument('--label_type', type=str, default='last_timestep', choices=['last_timestep', 'all_timesteps'],
                    help='how labels are fed into net: last_timestep or all_timesteps')
parser.add_argument('--layer_type', type=str, default='LstmC', choices=['LstmC', 'LstmNet'],
                    help='type of e-prop derivation: LstmC or LstmNet')
parser.add_argument('--single_exp', type=str2bool, nargs='?',
                    const=True, default=False,
                    help="activate single experiment mode")
parser.add_argument('--val_split', type=float, default=.1,
                    help='split for train and val data, positive float in (0,1)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=300,
                    help='upper epoch limit')
parser.add_argument('--runs', type=int, default=7,
                    help='repeat evaluation experiments')
parser.add_argument('--batch_size', type=int, default=32, choices=range(1, N + 1),
                    help='batch size')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
args = parser.parse_args()

current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../')

###############################################################################
# Set Up
###############################################################################
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

###############################################################################
# Load data
###############################################################################
print('LOADING DATA')
data_path = os.path.abspath(current_dir + '/dat')

TRAINING_DATA = torch.load(data_path + '/training_data.pt')
TRAINING_LABELS = torch.load(data_path + '/training_labels.pt')
TRAINING_LENGTHS = torch.load(data_path + '/training_lengths.pt')  # split TRAINING and VALIDATION
TEST_DATA = torch.load(data_path + '/test_data.pt')
TEST_LABELS = torch.load(data_path + '/test_labels.pt')
TEST_LENGTHS = torch.load(data_path + '/test_lengths.pt')

TEST_DATA, TEST_LABELS, TEST_LENGTHS = TEST_DATA.to(device), TEST_LABELS.to(device), TEST_LENGTHS.to(device)

n_train = TRAINING_DATA.size(0)
idx = np.arange(n_train)
val_split = int(np.floor(args.val_split * n_train))
train_indices, val_indices = idx[val_split:], idx[:val_split]

# Creating PT data samplers and loaders:
train_sampler = data.SubsetRandomSampler(train_indices)
val_sampler = data.SubsetRandomSampler(val_indices)

train_dataset = CustomDataset(TRAINING_DATA[train_indices], TRAINING_LABELS[train_indices],
                              TRAINING_LENGTHS[train_indices])
val_dataset = CustomDataset(TRAINING_DATA[val_indices], TRAINING_LABELS[val_indices], TRAINING_LENGTHS[val_indices])

train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size)
val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)
print('DATA LOADED')


###############################################################################
# BPTT LAST TIMESTEP LABEL
###############################################################################

def last_timestep_bptt():
    # test accs from each run
    losses = []
    accs = []

    path = '/char/{}/bptt'.format(args.label_type)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeat_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr.', 'test_loss', 'test_accuracy'])
        for i in range(args.runs):
            best_acc = 0.
            model = LstmBpttCharLastTimestep(input_size=5, hidden_size=args.nhid, output_size=20, batch_first=True).to(
                device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # start with training
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'elapsed_time'])
                for epoch in range(args.epochs):
                    model.train()
                    start_time = time.time()

                    train_loss, train_acc = 0., 0.
                    nbatch_train = len(train_loader)

                    for batch, labels, lengths in train_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        # clear out the gradients
                        optimizer.zero_grad()

                        # Run forward pass
                        ys = model(batch, lengths)

                        # Compute loss, gradients and update parameters by calling optimizer.step()
                        loss = criterion(ys, labels)

                        loss.backward()
                        # print(model.hidden2tag.bias.grad)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                        #
                        # update train loss
                        train_loss += loss.item()
                        # update train acc
                        label_pred = torch.argmax(ys, dim=-1)
                        acc = accuracy_score(label_pred, labels)
                        train_acc += acc
                    train_loss /= nbatch_train
                    train_acc /= nbatch_train

                    # evaluation
                    model.eval()
                    val_loss, val_acc = 0., 0.
                    nbatch_val = len(val_loader)
                    for batch, labels, lengths in val_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        y_pred = model(batch, lengths)
                        # labels = torch.repeat_interleave(labels, lengths, dim=0)
                        # update val loss
                        loss = criterion(y_pred, labels)
                        val_loss += loss.item()

                        # update val acc
                        label_pred = torch.argmax(y_pred, dim=-1)
                        acc = accuracy_score(label_pred, labels)
                        val_acc += acc
                    val_loss /= nbatch_val
                    val_acc /= nbatch_val

                    # save model with the current best acc
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
                    elapsed_time = round(time.time() - start_time, 2)
                    # print('Epoch {}/{} \t train loss={:.4f} \t train accuracy={:.2f}% \t val loss={:.4f} \t val
                    # accuracy={' ':.2f}%  \t time={:.2f}s \n'.format( epoch + 1, args.epochs, train_loss,
                    # train_acc * 100, val_loss, val_acc * 100, elapsed_time))
                    writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed_time])
            # testing
            model.load_state_dict(torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                ys = model(TEST_DATA, TEST_LENGTHS)
                loss = criterion(ys, TEST_LABELS)
                losses += [loss.item()]
                label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
                acc = accuracy_score(label_pred, TEST_LABELS)
                accs += [acc]
            print('test loss = {:.6f} \t test accuracy = {:.2f}%'.format(loss, acc * 100))
            writer_eval.writerow([i, loss.item(), acc])

        losses_mean = np.mean(losses)
        losses_std = np.std(losses)

        acc_mean = np.mean(accs)
        acc_std = np.std(accs)

        print('mean loss = {:.6f} \t std loss = {:.6f} \t mean acc = {:.2f}% \t std acc = {:.2f}%'.format(losses_mean,
                                                                                                          losses_std,
                                                                                                          acc_mean * 100,
                                                                                                          acc_std * 100))
        writer_eval.writerow(['mean', losses_mean, acc_mean])
        writer_eval.writerow(['std', losses_std, acc_std])
    return losses, accs, losses_mean, acc_mean, losses_std, acc_std


###############################################################################
# BPTT ALL TIMESTEPS LABEL
###############################################################################
def all_timesteps_bptt():
    # test accs from each run
    losses = []
    accs = []
    path = '/char/{}/bptt'.format(args.label_type)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeat_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr.', 'test_loss', 'test_accuracy'])
        for i in range(args.runs):
            best_acc = 0.
            model = LstmBpttCharAllTimesteps(input_size=5, hidden_size=args.nhid, output_size=20, batch_first=True).to(
                device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # start with training
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'elapsed_time'])
                for epoch in range(args.epochs):
                    model.train()
                    start_time = time.time()

                    train_loss, train_acc = 0., 0.
                    nbatch_train = len(train_loader)

                    for batch, labels, lengths in train_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        labels = torch.repeat_interleave(labels, lengths, dim=0)
                        # clear out the gradients
                        optimizer.zero_grad()

                        # Run forward pass
                        ys = model(batch, lengths)

                        # Compute loss, gradients and update parameters by calling optimizer.step()
                        loss = criterion(ys, labels)

                        loss.backward()
                        # print(model.hidden2tag.bias.grad)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                        optimizer.step()
                        #
                        # update train loss
                        train_loss += loss.item()
                        # update train acc
                        label_pred = torch.argmax(ys, dim=-1)
                        acc = accuracy_score(label_pred, labels)
                        train_acc += acc
                    train_loss /= nbatch_train
                    train_acc /= nbatch_train

                    # evaluation
                    model.eval()
                    val_loss, val_acc = 0., 0.
                    nbatch_val = len(val_loader)
                    for batch, labels, lengths in val_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        y_pred = model(batch, lengths)
                        labels = torch.repeat_interleave(labels, lengths, dim=0)
                        # update val loss
                        loss = criterion(y_pred, labels)
                        val_loss += loss.item()

                        # update val acc
                        label_pred = torch.argmax(y_pred, dim=-1)
                        acc = accuracy_score(label_pred, labels)
                        val_acc += acc
                    val_loss /= nbatch_val
                    val_acc /= nbatch_val

                    # save model with the current best acc
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))
                    elapsed_time = round(time.time() - start_time, 2)
                    # print('Epoch {}/{} \t train loss={:.4f} \t train accuracy={:.2f}% \t val loss={:.4f} \t val
                    # accuracy={' ':.2f}%  \t time={:.2f}s \n'.format( epoch + 1, args.epochs, train_loss,
                    # train_acc * 100, val_loss, val_acc * 100, elapsed_time))
                    writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed_time])
            # testing
            model.load_state_dict(torch.load(model_path + '/model_{}.pt'.format(i), map_location=torch.device(device)))
            with torch.no_grad():
                test_labels = torch.repeat_interleave(TEST_LABELS, TEST_LENGTHS,
                                                      dim=0)  # repeat labels w.r.t. seq lengths
                ys = model(TEST_DATA, TEST_LENGTHS)
                loss = criterion(ys, test_labels)
                losses += [loss.item()]
                label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
                acc = accuracy_score(label_pred, test_labels)
                accs += [acc]
            print('test loss = {:.6f} \t test accuracy = {:.2f}%'.format(loss, acc * 100))
            writer_eval.writerow([i, loss.item(), acc])

        losses_mean = np.mean(losses)
        losses_std = np.std(losses)

        acc_mean = np.mean(accs)
        acc_std = np.std(accs)

        print('mean loss = {:.6f} \t std loss = {:.6f} \t mean acc = {:.2f}% \t std acc = {:.2f}%'.format(losses_mean,
                                                                                                          losses_std,
                                                                                                          acc_mean * 100,
                                                                                                          acc_std * 100))
        writer_eval.writerow(['mean', losses_mean, acc_mean])
        writer_eval.writerow(['std', losses_std, acc_std])
    return losses, accs, losses_mean, acc_mean, losses_std, acc_std


'''Now only for last time step'''


###############################################################################
# E-PROP1 LAST TIME STEP LABEL
###############################################################################

def last_timestep_eprop1():
    def init_model():
        mdl = LstmEprop1LastTimestep(globals()[args.layer_type], input_size=5, hidden_size=args.nhid,
                                     output_size=20,
                                     batch_first=True, feedback_type=args.feedback_type).to(device)
        return mdl

    accs = []
    losses = []
    path = '/char/{}/eprop1/{}/{}'.format(args.label_type, args.feedback_type, args.layer_type)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeat_eval_{}.csv'.format(args.layer_type), 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr.', 'test_loss', 'test_accuracy'])
        for i in range(args.runs):
            best_acc = 0.
            model = init_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # disable backprop
            for param in model.parameters():
                param.requires_grad = False
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'elapsed_time'])
                for epoch in range(args.epochs):
                    start_time = time.time()
                    model.train()
                    train_loss, train_acc = 0., 0.
                    nbatch_train = len(train_loader)
                    for batch, labels, lengths in train_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        # clear grads
                        optimizer.zero_grad()

                        ys, lstm_output_T, _, et = model(batch, lengths)  # tag scores [seq*  B x O]
                        loss = criterion(ys, labels)

                        # calculate and assign gradients
                        model.eprop_cross_entropy(ys, lstm_output_T, et, labels)

                        # update weights
                        optimizer.step()

                        # update train loss
                        train_loss += loss.item()
                        # update train acc
                        label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
                        acc = accuracy_score(label_pred, labels)
                        train_acc += acc
                    train_loss /= nbatch_train
                    train_acc /= nbatch_train

                    model.eval()
                    val_loss, val_acc = 0., 0.
                    nbatch_val = len(val_loader)

                    for batch, labels, lengths in val_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        batch_size = batch.size(0)
                        state = init_hidden(batch_size, model.hidden_size)
                        y_pred, _ = model.forward_eval(batch, lengths, state)

                        # update val loss
                        loss = criterion(y_pred, labels)
                        val_loss += loss.item()

                        # update val acc
                        label_pred = torch.argmax(y_pred, dim=-1)
                        acc = accuracy_score(label_pred, labels)
                        val_acc += acc
                    val_loss /= nbatch_val
                    val_acc /= nbatch_val

                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))

                    elapsed_time = round(time.time() - start_time, 2)
                    # print('Epoch {}/{} \t train loss={:.4f} \t train accuracy={:.2f}% \t val loss={:.4f} \t val
                    # accuracy={' ':.2f}%  \t time={:.2f}s \n'.format( epoch + 1, args.epochs, train_loss,
                    # train_acc * 100, val_loss, val_acc * 100, elapsed_time))
                    writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed_time])
            # testing
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i),
                           map_location=torch.device(device)))

            state = init_hidden(TEST_DATA.size(0), model.hidden_size)
            ys, _ = model.forward_eval(TEST_DATA, TEST_LENGTHS, state)
            loss = criterion(ys, TEST_LABELS)
            losses += [loss.item()]
            label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
            acc = accuracy_score(label_pred, TEST_LABELS)
            accs += [acc]
            print('test loss = {:.6f} \t test accuracy = {:.2f}%'.format(loss, acc * 100))
            writer_eval.writerow([i, loss.item(), acc])

        losses_mean = np.mean(losses)
        losses_std = np.std(losses)

        acc_mean = np.mean(accs)
        acc_std = np.std(accs)

        print('mean loss = {:.6f} \t std loss = {:.6f} \t mean acc = {:.2f}% \t std acc = {:.2f}%'.format(losses_mean,
                                                                                                          losses_std,
                                                                                                          acc_mean * 100,
                                                                                                          acc_std * 100))
        writer_eval.writerow(['mean', losses_mean, acc_mean])
        writer_eval.writerow(['std', losses_std, acc_std])
    return losses, accs, losses_mean, acc_mean, losses_std, acc_std


###############################################################################
# E-PROP1 ALL TIMESTEPS LABEL
###############################################################################
def all_timesteps_eprop1():
    def init_model():
        mdl = LstmEprop1AllTimesteps(globals()[args.layer_type], input_size=5, hidden_size=args.nhid, output_size=20,
                                     batch_first=True, feedback_type=args.feedback_type).to(device)
        return mdl

    accs = []
    losses = []
    path = '/char/{}/eprop1/{}/{}'.format(args.label_type, args.feedback_type, args.layer_type)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    with open(exp_path + '/repeat_eval.csv', 'w+') as f_eval:
        writer_eval = csv.writer(f_eval)
        writer_eval.writerow(['run_Nr.', 'test_loss', 'test_accuracy'])
        for i in range(args.runs):
            best_acc = 0.
            model = init_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # disable backprop
            for param in model.parameters():
                param.requires_grad = False
            with open(res_path + '/res_{}.csv'.format(i), 'w+') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'elapsed_time'])
                for epoch in range(args.epochs):
                    start_time = time.time()
                    model.train()
                    train_loss, train_acc = 0., 0.
                    nbatch_train = len(train_loader)
                    for batch, labels, lengths in train_loader:
                        seq = batch.size(1)
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        labels_seqs = labels.repeat(seq)
                        # clear grads
                        optimizer.zero_grad()

                        ys, lstm_outputs, _, ets, _ = model(batch)  # tag scores [seq*  B x O]
                        loss = criterion(ys, labels_seqs)

                        # calculate and assign gradients
                        model.eprop_cross_entropy(ys, lstm_outputs, ets, labels_seqs)

                        # update weights
                        optimizer.step()

                        # update train loss
                        train_loss += loss.item()
                        # update train acc
                        label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
                        acc = accuracy_score(label_pred, labels_seqs)
                        train_acc += acc
                    train_loss /= nbatch_train
                    train_acc /= nbatch_train

                    model.eval()
                    val_loss, val_acc = 0., 0.
                    nbatch_val = len(val_loader)

                    for batch, labels, lengths in val_loader:
                        batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                        batch_size, seq, _ = batch.size()
                        state = init_hidden(batch_size, model.hidden_size)
                        labels_seqs = labels.repeat(seq)
                        y_pred, _ = model.forward_eval(batch, state)

                        # update val loss
                        loss = criterion(y_pred, labels_seqs)
                        val_loss += loss.item()

                        # update val acc
                        label_pred = torch.argmax(y_pred, dim=-1)
                        acc = accuracy_score(label_pred, labels_seqs)
                        val_acc += acc
                    val_loss /= nbatch_val
                    val_acc /= nbatch_val

                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), model_path + '/model_{}.pt'.format(i))

                    elapsed_time = round(time.time() - start_time, 2)
                    # print('Epoch {}/{} \t train loss={:.4f} \t train accuracy={:.2f}% \t val loss={:.4f} \t val
                    # accuracy={' ':.2f}%  \t time={:.2f}s \n'.format( epoch + 1, args.epochs, train_loss,
                    # train_acc * 100, val_loss, val_acc * 100, elapsed_time))
                    writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed_time])
            # testing
            model.load_state_dict(
                torch.load(model_path + '/model_{}.pt'.format(i),
                           map_location=torch.device(device)))

            state = init_hidden(TEST_DATA.size(0), model.hidden_size)
            ys, _ = model.forward_eval(TEST_DATA, state)
            seq = TEST_DATA.size(1)
            test_labels = TEST_LABELS.repeat(seq)
            loss = criterion(ys, test_labels)
            losses += [loss.item()]
            label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
            acc = accuracy_score(label_pred, test_labels)
            accs += [acc]
            print('test loss = {:.6f} \t test accuracy = {:.2f}%'.format(loss, acc * 100))
            writer_eval.writerow([i, loss.item(), acc])

        losses_mean = np.mean(losses)
        losses_std = np.std(losses)

        acc_mean = np.mean(accs)
        acc_std = np.std(accs)

        print('mean loss = {:.6f} \t std loss = {:.6f} \t mean acc = {:.2f}% \t std acc = {:.2f}%'.format(losses_mean,
                                                                                                          losses_std,
                                                                                                          acc_mean * 100,
                                                                                                          acc_std * 100))
        writer_eval.writerow(['mean', losses_mean, acc_mean])
        writer_eval.writerow(['std', losses_std, acc_std])
    return losses, accs, losses_mean, acc_mean, losses_std, acc_std


def single_exp_all_timesteps_eprop1():
    path = '/char/single_exp/{}/eprop1/{}/{}'.format(args.label_type, args.feedback_type, args.layer_type)
    exp_path = current_dir + path
    res_path = exp_path + '/results'
    model_path = exp_path + '/models'

    create_folder_when_necessary(res_path)
    create_folder_when_necessary(model_path)
    create_folder_when_necessary(exp_path)

    best_acc = 0.
    model = LstmEprop1AllTimesteps(globals()[args.layer_type], input_size=5, hidden_size=args.nhid,
                                   output_size=20,
                                   batch_first=True, feedback_type=args.feedback_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # switch-off backprop
    def enable_autograd(is_using_bptt):
        for param in model.parameters():
            param.requires_grad = is_using_bptt

    # initialization, only True when need information from bptt, otherwise False for eprop
    enable_autograd(False)

    checkpoints = [int(1 / 5 * args.epochs), int(1 / 2 * args.epochs), args.epochs - 1]
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
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'elapsed_time'])
        for epoch in range(args.epochs):
            model.train()
            train_loss, train_acc = 0., 0.
            nbatch_train = len(train_loader)
            for i, (batch, labels, lengths) in enumerate(train_loader):
                seq = batch.size(1)
                batch, labels, lengths = batch.to(device), labels.to(device), lengths.to(device)
                labels_seqs = labels.repeat(seq)
                # clear grads
                optimizer.zero_grad()

                ys, lstm_outputs, state, ets, caches = model(batch)  # tag scores [seq*  B x O]
                loss = criterion(ys, labels_seqs)

                if i==0 and epoch in checkpoints:
                    Ls_eprop = model.eprop_cross_entropy(ys, lstm_outputs, ets, labels_seqs)
                    ets_cpu = tuple([et.to('cpu') for et in ets])
                    eprop_ls_checkpoints += [Ls_eprop.to('cpu')]
                    save_ets_checkpoints(ets_cpu)
                    save_grad_checkpoints(eprop_grad_checkpoints, Ls_eprop.to('cpu'), ets_cpu)
                    # rerun backprop
                    optimizer.zero_grad()
                    Ls_bptt = model.backward_cross_entropy(ys, lstm_outputs, caches, labels_seqs, batch)
                    bptt_ls_checkpoints += [Ls_bptt.to('cpu')]
                    save_grad_checkpoints(bptt_grad_checkpoints, Ls_bptt.to('cpu'), ets_cpu)
                    optimizer.zero_grad()

                model.eprop_cross_entropy(ys, lstm_outputs, ets, labels_seqs)
                optimizer.step()

                # update train loss
                train_loss += loss.item()
                # update train acc
                label_pred = torch.argmax(ys, dim=-1)  # tag scores [seq x B x O]
                acc = accuracy_score(label_pred, labels_seqs)
                train_acc += acc
            train_loss /= nbatch_train
            train_acc /= nbatch_train
            print('Epoch {}/{} \t train loss={:.4f} \t train accuracy={:.2f}% \n'.format(epoch + 1, args.epochs,
                                                                                         train_loss,
                                                                                         train_acc * 100))

    checkpints_path = exp_path + '/checkpoints'
    create_folder_when_necessary(checkpints_path)
    torch.save(ets_checkpoints, checkpints_path + '/ets_checkpoints.pt')
    torch.save(eprop_grad_checkpoints, checkpints_path + '/eprop_grad_checkpoints.pt')
    torch.save(bptt_grad_checkpoints, checkpints_path + '/bptt_grad_checkpoints.pt')
    torch.save(eprop_ls_checkpoints, checkpints_path + '/eprop_ls_checkpoints.pt')
    torch.save(bptt_ls_checkpoints, checkpints_path + '/bptt_ls_checkpoints.pt')



current_dir += '/test'

if __name__ == "__main__":
    if args.single_exp:
        func = globals()['single_exp_all_timesteps_eprop1']
    else:
        func = globals()[args.label_type + '_' + args.model]
    func()
