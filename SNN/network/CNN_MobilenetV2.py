import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
__all__ = ["mobilenetv2", "MobileNetV2"]

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    # stride 支持 int 或 (sh, sw)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
# #-------------------------------------------------------------------
# class CR(nn.Module) : 
#     def __init__(self, n_band, overlap=1/3, **kwargs):
#         super(CR, self).__init__()
#         self.n_band = n_band
#         self.overlap = overlap
#         """
#         if type_window == "None" :
#             self.window = torch.tensor(1.0)
#         elif type_window == "Rectengular" : 
#             self.window = torch.kaiser_window(window_length ,beta = 0.0)
#         elif type_window == "Hanning":
#             self.window = torch.hann_window(window_length)
#         else :
#             raise NotImplementedError
#         """

#     def forward(self,x):
#         idx = 0

#         B,C,T,F = x.shape  ##  2,1,63,257
#         n_freq = x.shape[3]
#         sz_band = n_freq/(self.n_band*(1-self.overlap))
#         sz_band = int(np.ceil(sz_band))
#         y = torch.zeros(B,self.n_band*C,T,sz_band).to(x.device)
#         for i in range(self.n_band):
#             if idx+sz_band > F :
#                 sz_band = F - idx
#             y[:,i*C:(i+1)*C,:,:sz_band] = x[:,:,:,idx:idx+sz_band]
#             n_idx = idx + int(sz_band*(1-self.overlap))
#             idx = n_idx
#         return y
# #-------------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        # stride 可为 1/2 或 (2,1)
        if isinstance(stride, int):
            assert stride in [1, 2]
            stride_is_one = (stride == 1)
        else:
            assert stride[0] in [1, 2] and stride[1] in [1, 2]
            stride_is_one = (stride[0] == 1 and stride[1] == 1)

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride_is_one and (inp == oup)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    """
    输入:  (N, 1, 1498, 1024)
    输出:  (N, 9, 375)
    通过 keep_width=True 在宽度方向不做下采样（stride=(2,1)）
    """
    def __init__(self, out_channels=9, out_steps=None, in_channels=1, width_mult=1., keep_width=True):
        super().__init__()
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.keep_width = keep_width
        # self.CR = CR(8,overlap=1/5)

        # t(扩展倍数), c(输出通道), n(重复次数), s(步幅)
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]

        # 第一层
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        first_stride = (1, 2) if keep_width else 2  # 只在高度下采样
        layers = [conv_3x3_bn(in_channels, input_channel, first_stride)]

        # 倒残差块
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            stride_val = (2, 2) if (self.keep_width and s == 2) else s
            for i in range(n):
                layers.append(block(input_channel, output_channel, stride_val if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # 最后 1x1 conv
        last_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, last_channel)

        # 序列头：H -> 1，W -> out_steps；然后 1×1 conv 到 out_channels
        self.pool = nn.AdaptiveAvgPool2d((out_steps,1))
        self.proj = nn.Conv2d(last_channel, out_channels, kernel_size=1, bias=True)

        self._initialize_weights()

    def forward(self, x):
        # x = self.CR(x)         #[2, 8, 80, 160]
        # print(f"x",x.shape)
        x = self.features(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), ceil_mode=True)
        x = self.conv(x)     # (N, C, H', W'≈1024)
        
        x = self.pool(x)     # (N, C, 1, out_steps=375)
        x = self.proj(x)     # (N, 9, 1, 375)
        # print(f"x",x.shape)
        x = x.squeeze(3)     # (N, 9, 375)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    return MobileNetV2(**kwargs)

# ----------------- 简单自测 -----------------
# if __name__ == "__main__":
#     device=torch.device("cpu")
#     KWS = MobileNetV2().to(device)
#     model = mobilenetv2(out_channels=9, out_steps=None, in_channels=1, keep_width=True)
#     x = torch.randn(1, 1, 1498, 1024)
#     y = model(x)
#     print("输出形状:", y.shape)  # 期望: torch.Size([1, 9, 20])
#     torch.save(KWS.state_dict(), 'test.model')

#     # 统计参数量
#     total_params = sum(p.numel() for p in KWS.parameters() if p.requires_grad)
#     print(f"模型参数量: {total_params:,}")

#     # 计算模型大致大小（float32）
#     model_size_MB = total_params * 4 / 1024 / 1024
#     print(f"模型大小约为: {model_size_MB:.2f} MB (float32精度)")

#     # 推荐：详细结构和参数统计（需要 pip install torchinfo）
#     try:
#         from torchinfo import summary
#         print("\n=== 模型结构和每层参数量统计 ===")
#         summary(KWS, input_size=(1, 1, 1498, 1024))
#     except ImportError:
#         print("\n未安装 torchinfo，跳过详细结构统计。可用 pip install torchinfo 安装。")
#         # # 可选：统计FLOPs（需要 pip install thop）
#     try:
#         from thop import profile
#         device=torch.device("cpu")
#         x = torch.randn(1, 1, 1498, 1024,device=device)
#         dummy_input=torch.randn(x.shape)
#         model_no_cr = MobileNetV2()
#         macs, params = profile(model_no_cr, inputs=(dummy_input, ))
#         print(f"\nFLOPs: {macs}, 参数量（再统计一遍）: {params}")
#     except ImportError:
#         print("\n未安装 thop，跳过FLOPs统计。可用 pip install thop 安装。")
