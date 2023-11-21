import torch
import torch.nn as nn


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

    def forward(self, x, mask=None):
        # Self_attention part (use self.norm1)
        multi_head_attn, _ = self.self_attn.forward(x, x, x, attn_mask=mask, need_weights=False)
        multi_head_attn = self.dropout(multi_head_attn)
        normed = self.norm1(x + multi_head_attn)

        # MLP part (use self.norm2)
        ffn_out = self.linear_net.forward(normed)
        ffn_out = self.dropout(ffn_out)
        normed = self.norm2(x + ffn_out)

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

    def forward(self, x, memory, src_mask, tgt_mask):
        # Self-Attention part (use self.norm1)

        masked_attn, *_ = self.self_attn.forward(x, x, x, attn_mask=tgt_mask, need_weights=False)
        masked_attn = self.dropout(masked_attn)
        normed = self.norm1(x + masked_attn)

        # Attention part (use self.norm2)
        # Recall that memory is the output of the encoder and replaces x as
        # the key and value in the attention layer

        attn, *_ = self.src_attn.forward(normed, memory, memory, attn_mask=src_mask, need_weights=False)
        attn = self.dropout(attn)
        normed = self.norm2(normed + attn)

        # MLP part (use self.norm3)
        linear_out = self.linear_net.forward(normed)
        linear_out = self.dropout(linear_out)
        x = self.norm3(normed + linear_out)

        return x


if __name__ == "__main__":
    data = torch.randn((8, 25, 22 * 3))
    target = torch.randn((8, 25, 22 * 3))

    enc = EncoderBlock(input_dim=22*3, num_heads=6, dim_feedforward=64, dropout=0.1)

    out = enc(data)

    dec = DecoderBlock(input_dim=22*3, num_heads=6, dim_feedforward=64, dropout=0.1)

    out = dec.forward(x=target, memory=out, src_mask=None, tgt_mask=None)
    out

