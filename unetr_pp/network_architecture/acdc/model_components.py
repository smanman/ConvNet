from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.acdc.transformerblock import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

import torch
einops, _ = optional_import("einops")
class MultiPathAttn(nn.Module):
    def __init__(self, channels, reduction_ratio=16, num_paths=4):
        super(MultiPathAttn, self).__init__()

        self.num_paths = num_paths

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=(1,1,1)),
        )
        # print("Shared MLP weight shape:", list(self.shared_mlp.parameters())[0].shape)

        self.path_mlp = nn.ModuleList()
        for _ in range(num_paths):
            self.path_mlp.append(nn.Sequential(
                nn.Conv3d(channels, channels // reduction_ratio, kernel_size=(1,1,1)),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels // reduction_ratio, channels, kernel_size=(1,1,1)),
            ))
            # print("Path", _+1, "MLP weight shape:", list(self.path_mlp[-1].parameters())[0].shape)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.fc = nn.Sequential(
            nn.Conv3d(channels * 10, channels * 5, kernel_size=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels * 5, channels, kernel_size=(1,1,1)),
        )
        # print("FC weight shape:", list(self.fc.parameters())[0].shape)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, channels, depth, height, width = x.size()

        # Shared MLP
        shared_out = self.shared_mlp(x)

        # Path-specific MLPs
        path_out = []
        for i in range(self.num_paths):
            path_out.append(self.path_mlp[i](x))

        # Pooling and concatenation
        avg_out = self.avgpool(shared_out)
        max_out = self.maxpool(shared_out)

        # Concatenate pooled features from all paths
        for i in range(self.num_paths):
            avg_out = torch.cat([avg_out, self.avgpool(path_out[i])], dim=1)
            max_out = torch.cat([max_out, self.maxpool(path_out[i])], dim=1)

        # Channel-wise attention
        a = torch.cat([avg_out, max_out], dim=1)
        channel_attention = self.sigmoid(self.fc(a))
        # print("Channel attention weight shape:", list(self.fc.parameters())[0].shape)

        # Spatial attention
        spatial_attention = torch.sigmoid(torch.mean(shared_out, dim=1, keepdim=True))

        # Apply attention
        out = channel_attention * shared_out * spatial_attention

        return out
class CBAMLayer(nn.Module):
    def __init__(self, channel, spatial_kernel=3):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W,D为1
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.reduction =channel // 2
        # shared MLP
        torch.Size([1, 128, 4, 10, 10])
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            #nn.Linear(channel, int(channel *0.5), bias=False),
            nn.Conv3d(channel, int(channel *0.5), 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            #nn.Linear(int(channel *0.5), channel,bias=False)
            nn.Conv3d(int(channel *0.5), channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv3d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # print(self.max_pool(x).shape)
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out)
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        a = self.conv(torch.cat([max_out, avg_out], dim=1))

        spatial_out = self.sigmoid(a)

        x = spatial_out * x
        return x
class SCS3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCS3D, self).__init__()

        self.conv1 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, stride=1,padding=1)
        self.conv2_reduce = nn.Conv3d(out_channels, out_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 16)
        self.fc2 = nn.Linear(out_channels // 16, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        out_x = torch.cat([x1, x2], dim=1) #out:[1,256,4,10,10]
        out = self.bn1(self.conv1(out_x)) # [1,128,4,10,10]
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        squeeze = self.avgpool(out)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).unsqueeze(2).unsqueeze(3).unsqueeze(4)  #权重
        out = x1 * excitation.expand_as(out)

        out = out + x2
        return out
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

# ConvNeXt Block
class ux_block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = x .permute(0, 4, 1, 2, 3)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)
        x = input + self.drop_path(x)
        return x

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],dims=[32, 64, 128, 256],
                 proj_size =[64,64,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.15 ,drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3] ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # 嵌入层 embedding
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        self.cbam=CBAMLayer(256) #跳跃融合层  ###标记
        self.cbam_change = MultiPathAttn(256)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.skips= nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            # skip_blocks=[]
            if i == 0:
                for j in range(depths[i]):
                    stage_blocks.append(
                        TransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i],
                                         num_heads=num_heads,
                                         dropout_rate=transformer_dropout_rate, pos_embed=True))
            else:
                for j in range(depths[i]):
                    stage_blocks.append(ux_block(dim=dims[i], drop_path=dp_rates[cur + j],
                                                 layer_scale_init_value=layer_scale_init_value))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward_features(self, x):
        # x: [1, 1, 16, 160, 160] [1, 32, 8, 40, 40] [64, 16, 16, 16] [128, 8, 8, 8] [256, 4, 4, 4]
        outs = []
        for i in range(4):
            # print(i)
            # print(x.size())
            x = self.downsample_layers[i](x)
            # print(x.size())
            x = self.stages[i](x)
            # x=self.skips[i](x)
            if i==3:
                x = self.cbam(x)
            else:
                pass
            # x = self.cbam(x)
            # print(x.size())
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        hidden_states = self.forward_features(x)
        return hidden_states


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()
        # self.cbam=CBAMLayer(out_channels)
        self.scs3D=SCS3D(in_channels,out_channels)
        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        # 解码器上采样特征
        out = self.transp_conv(inp)

        # 使用LHA模块来融合高低层特征
        out = self.lha(out, skip)

        return out
