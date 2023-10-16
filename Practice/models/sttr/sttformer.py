import torch.nn as nn
from .sta_block import STA_Block


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class Model(nn.Module):
    def __init__(self, num_joints, 
                 num_frames, num_frames_out, num_heads, num_channels, 
                 kernel_size, len_parts=1, use_pes=True, config=None, num_persons=1,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        config = [[16,16,  16], [16,16,  16], 
            [16,16,  16], [16,16,  16],
            [16,  16,  16], [16,16,  16], 
            [16,16,  16], [16,3,  16]]

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_persons = num_persons
        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]

        num_frames = num_frames // len_parts
        num_joints = num_joints * len_parts
        
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(STA_Block(in_channels, out_channels, qkv_dim, 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))   
        self.fc_out = nn.Linear(66, 66)
        self.conv_out = nn.Conv1d(num_frames, num_frames_out, 1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels, self.num_persons).permute(0, 3, 1, 2, 4).contiguous()
        N, C, T, V, M = x.shape
        
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        x = self.input_map(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)

            
        out = self.fc_out(self.conv_out(x.reshape(-1, self.num_frames, self.num_joints*self.num_channels)))  
        
        return out