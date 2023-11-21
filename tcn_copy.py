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
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None, src_key_padding_mask=None,
                              is_causal=None):
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
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                              tgt_key_padding_mask=None,
                              memory_key_padding_mask=None,
                              tgt_is_causal=None, memory_is_causal=None):
        return self.network(memory)


if __name__ == "__main__":
    data = torch.randn((256, 10, 96))
    target = torch.randn((256, 25, 96))
    enc = EncoderBlock(input_dim=96, num_heads=6, dim_feedforward=64, dropout=0.1)
    dec = DecoderBlock(input_dim=96, num_heads=6, dim_feedforward=64, dropout=0.1)
    out = enc(data)

    tcn = TemporalConvNet(10, [25])
    # num inputs is the number of frames, because these are my timesteps?
    #out = tcn.forward(data)

    tf = nn.Transformer(d_model=96, nhead=6, custom_encoder=enc, custom_decoder=tcn, batch_first=True)
    tf.forward(data, target)


