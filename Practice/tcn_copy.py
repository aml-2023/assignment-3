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

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, device=device)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward, device=device),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim, device=device)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim, device=device)
        self.norm2 = nn.LayerNorm(input_dim, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        # Self_attention part (use self.norm1)
        multi_head_attn, _ = self.self_attn.forward(src, src, src, attn_mask=mask, need_weights=False)
        multi_head_attn = self.dropout(multi_head_attn)
        normed = self.norm1(src + multi_head_attn)

        # MLP part (use self.norm2)
        ffn_out = self.linear_net.forward(normed)
        ffn_out = self.dropout(ffn_out)
        normed = self.norm2(src + ffn_out)

        x = normed

        return x


class DecoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Self Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        # Attention Layer
        self.src_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        # Self-Attention part (use self.norm1)

        masked_attn, *_ = self.self_attn.forward(tgt, tgt, tgt, attn_mask=tgt_mask, need_weights=False)
        masked_attn = self.dropout(masked_attn)
        normed = self.norm1(tgt + masked_attn)

        # Attention part (use self.norm2)
        # Recall that memory is the output of the encoder and replaces x as
        # the key and value in the attention layer

        attn, *_ = self.src_attn.forward(normed, memory, memory, attn_mask=memory_mask, need_weights=False)
        attn = self.dropout(attn)
        normed = self.norm2(normed + attn)

        # MLP part (use self.norm3)
        linear_out = self.linear_net.forward(normed)
        linear_out = self.dropout(linear_out)
        x = self.norm3(normed + linear_out)

        return x


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


if __name__ == "__main__":
    # # Arguments to setup the datasets
    datas = 'h36m'  # dataset name
    path = './data/h3.6m/h3.6m/dataset'
    input_n = 10  # number of frames to train on (default=10)
    output_n = 25  # number of frames to predict on
    input_dim = 3  # dimensions of the input coordinates(default=3)
    skip_rate = 1  # # skip rate of frames
    joints_to_consider = 22

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

    enc = EncoderBlock(input_dim=66, num_heads=6, dim_feedforward=64, dropout=0.1)

    # decoder is just a simple TCN, that takes in 10 channels (number of input frames) and predicts an output of 25
    dec = TemporalConvNet(10, [25, 25, 25])

    # num inputs is the number of frames, because these are my timesteps?
    tf = nn.Transformer(d_model=66, nhead=6, custom_encoder=enc, custom_decoder=dec, batch_first=True, device=device)

    train_tcn_transformer(tf, data_loader, vald_loader, device, 41)
