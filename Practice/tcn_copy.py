import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
import time
import pandas as pd
from torch.autograd import Variable

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, device=device))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, device=device))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, device=device) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        return self.network(memory)


def train_tcn_transformer(model, train_loader, val_loader, device, n_epochs, clip_grad=True, log_step=200):
    train_loss = []
    val_loss = []
    val_loss_best = 1000

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    use_scheduler = True  # use MultiStepLR scheduler

    if use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.1)

    joints_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                            46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                            75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    for epoch in range(n_epochs - 1):
        running_loss = 0
        n = 0
        model.train()

        for cnt, batch in enumerate(train_loader):
            batch = batch.float().to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            # the training sequences are the first 10 frames of the sequence with the joints that we are interested in
            sequences_train = batch[:, :input_n, joints_used]
            sequences_gt = batch[:, input_n:input_n + output_n, joints_used]

            pred = model.forward(sequences_train, sequences_gt)

            loss = mpjpe_error(pred, sequences_gt)

            if cnt % log_step == 0:
                print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' % (epoch + 1, cnt + 1, loss.item()))

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            running_loss += loss * batch_dim

        train_loss.append(running_loss.detach().cpu() / n)

        model.eval()

        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(val_loader):
                batch = batch.float().to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_train = batch[:, :input_n, joints_used]
                sequences_gt = batch[:, input_n:input_n + output_n, joints_used]

                sequences_predict = model.forward(sequences_train, sequences_gt)
                loss = mpjpe_error(sequences_predict, sequences_gt)

                if cnt % log_step == 0:
                    print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' % (epoch + 1, cnt + 1, loss.item()))

                running_loss += loss * batch_dim

            val_loss.append(running_loss.detach().cpu() / n)

            if running_loss / n < val_loss_best:
                val_loss_best = running_loss / n

        if use_scheduler:
            scheduler.step()


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)

        # --------- start of our code

        # all possible positions
        positions = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)

        # getting the even dimensions
        even_dims = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)

        # getting the odd dimensions
        odd_dims = torch.arange(1, d_model, 2, dtype=torch.float32, device=device)

        # calculating the denominators
        sin_denominator = torch.pow(10000.0, even_dims / d_model)
        cos_denominator = torch.pow(10000.0, (odd_dims - 1) / d_model)

        # assigning the sin output to the even dimensions
        pe[:, 0::2] = torch.sin(positions / sin_denominator)

        # assigning the cos output to the odd dimensions
        pe[:, 1::2] = torch.cos(positions / cos_denominator)

        # ------- end of our code

        pe = pe.unsqueeze(0)  # the final dimension is (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class CustomModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = 512
        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=0.1)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=dim_feed_forward)
        enc = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # decoder is just a simple TCN, that takes in 10 channels (number of input frames) and predicts an output of 25
        dec = TemporalConvNet(10, [25, 25, 25])

        self.embed = nn.Linear(66, d_model, device=device)

        self.tf = nn.Transformer(d_model=d_model, nhead=8, custom_encoder=enc, custom_decoder=dec, batch_first=True,
                                 device=device)

    def forward(self, src, target):
        src_embed = self.embed.forward(src)
        src_encoded = self.pos_enc.forward(src_embed)
        target_encoded = self.pos_enc.forward(target)
        return self.tf.forward(src_encoded, target_encoded)


if __name__ == "__main__":
    # # Arguments to setup the datasets
    datas = 'h36m'  # dataset name
    path = './data/h3.6m/h3.6m/dataset'
    input_n = 10  # number of frames to train on (default=10)
    output_n = 25  # number of frames to predict on
    input_dim = 3  # dimensions of the input coordinates(default=3)
    skip_rate = 1  # # skip rate of frames
    joints_to_consider = 22

    d_model = 512
    dim_feed_forward = 2048
    num_enc_layers = 6
    # FLAGS FOR THE TRAINING
    mode = 'train'  # choose either train or test mode

    batch_size_test = 8
    model_path = './checkpoints/'  # path to the model checkpoint file

    actions_to_consider_test = 'all'  # actions to test on.
    model_name = datas + '_3d_' + str(output_n) + 'frames_ckpt'  # the model name to save/load

    # FLAGS FOR THE VISUALIZATION
    actions_to_consider_viz = 'all'  # actions to visualize
    visualize_from = 'test'
    n_viz = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load Data
    print('Loading Train Dataset...')
    dataset = datasets.Datasets(path, input_n, output_n, skip_rate, split=0)
    print('Loading Validation Dataset...')
    vald_dataset = datasets.Datasets(path, input_n, output_n, skip_rate, split=1)

    # ! Note: Ignore warning:  "VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences"
    batch_size = 256

    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)  #

    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = CustomModel()
    train_tcn_transformer(model, data_loader, vald_loader, device, 41)
