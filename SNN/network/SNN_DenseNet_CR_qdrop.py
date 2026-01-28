import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import numpy as np
#-------------------------------------------------------------------
decay = 0.25  # 0.25 # decay constants

class mem_update(nn.Module):
    def __init__(self, act: bool = False, p_init: float = 1.0):
        super().__init__()
        self.act = act
        self.qtrick = MultiSpike4(p=p_init)

    def forward(self, x):
        x = x.unsqueeze(0)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output.squeeze(0)

    
class MultiSpike4(nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.register_buffer("p_buf", torch.tensor(float(p)))

    @property
    def p(self) -> float:
        return float(self.p_buf.item())

    def set_p(self, p: float):
        self.p_buf.fill_(max(0.0, min(1.0, float(p))))

    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, 0.0, 4.0))
        @staticmethod
        def backward(ctx, grad_output):
            (inp,) = ctx.saved_tensors
            g = grad_output.clone()
            g[inp < 0.0] = 0.0
            g[inp > 4.0] = 0.0
            return g

    def forward(self, x):
        if not self.training:                # eval 阶段：全量化
            return self.quant4.apply(x)
        qv = self.quant4.apply(x)
        gate = torch.bernoulli(torch.full_like(x, self.p))  # 1=量化, 0=浮点
        out = x + (qv - x) * gate
        return torch.clamp(out, 0.0, 4.0)
class CR(nn.Module) : 
    def __init__(self, n_band, overlap=1/3, **kwargs):
        super(CR, self).__init__()
        self.n_band = n_band
        self.overlap = overlap
        """
        if type_window == "None" :
            self.window = torch.tensor(1.0)
        elif type_window == "Rectengular" : 
            self.window = torch.kaiser_window(window_length ,beta = 0.0)
        elif type_window == "Hanning":
            self.window = torch.hann_window(window_length)
        else :
            raise NotImplementedError
        """

    def forward(self,x):
        idx = 0

        B,C,T,F = x.shape  ##  2,1,63,257
        n_freq = x.shape[3]
        sz_band = n_freq/(self.n_band*(1-self.overlap))
        sz_band = int(np.ceil(sz_band))
        y = torch.zeros(B,self.n_band*C,T,sz_band).to(x.device)
        for i in range(self.n_band):
            if idx+sz_band > F :
                sz_band = F - idx
            y[:,i*C:(i+1)*C,:,:sz_band] = x[:,:,:,idx:idx+sz_band]
            n_idx = idx + int(sz_band*(1-self.overlap))
            idx = n_idx
        return y

# -----------------------------
# 1) DenseNet 基本单元：DenseLayer
# -----------------------------
class _DenseLayer(nn.Module):
    """
    DenseNet-BC 的一个层：
    BN-ReLU-1x1Conv -> BN-ReLU-3x3Conv
    输出会 concat 到输入特征上
    """
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.lif = mem_update()

        self.norm1 = nn.BatchNorm2d(in_channels)
        # self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        # self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.lif(self.norm1(x)))
        out = self.conv2(self.lif(self.norm2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        # Dense 连接：concat 输入与新特征
        out = torch.cat([x, out], dim=1)
        return out


# -----------------------------
# 2) DenseBlock：堆叠多个 DenseLayer
# -----------------------------
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer(c, growth_rate, bn_size=bn_size, drop_rate=drop_rate))
            c += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = c  # block 结束后的通道数

    def forward(self, x):
        return self.block(x)


# -----------------------------
# 3) Transition：压缩通道 + 下采样
# -----------------------------
class _Transition(nn.Module):
    """
    BN-ReLU-1x1Conv -> AvgPool2d(2,2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.lif = mem_update()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.lif(self.norm(x)))
        x = self.pool(x)
        return x


# -----------------------------
# 4) 手写 DenseNet Features 主干
# -----------------------------
class DenseNetFeatures(nn.Module):
    """
    DenseNet 的 features 部分（等价于 torchvision 的 net.features）
    参数基本对齐 DenseNet-BC：
    - growth_rate=32
    - bn_size=4
    - compression=0.5
    - block_config: 121/169/201/161 等配置
    """
    def __init__(
        self,
        in_channels=3,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        compression=0.5,
        drop_rate=0.0
    ):
        super().__init__()

        # Stem: conv0/norm0/relu0/pool0
        self.conv0 = nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        # self.relu0 = nn.ReLU(inplace=True)
        self.lif = mem_update()
        
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        c = num_init_features

        # DenseBlock + Transition (最后一个 block 后面没有 transition)
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, c, growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            self.blocks.append(block)
            c = block.out_channels

            if i != len(block_config) - 1:
                out_c = int(c * compression)
                trans = _Transition(c, out_c)
                self.transitions.append(trans)
                c = out_c

        # 最后 norm5
        self.norm5 = nn.BatchNorm2d(c)
        self.out_channels = c

    def forward(self, x):
        x = self.pool0(self.lif(self.norm0(self.conv0(x))))
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.norm5(x)
        return x


# -----------------------------
# 5) 你的 TF 输入 DenseNet（不封装版）
# -----------------------------
class DenseNetTF(nn.Module):
    """
    输入:  [B, Cin, T, F]
    内部:  permute -> [B, Cin, F, T]
    输出:  [B, num_classes, T']
    """
    def __init__(self, num_classes=9, in_channels=8, arch="densenet121", drop_rate=0.0):
        super().__init__()

        if arch == "densenet121":
            block_config = (6, 12, 24, 16)
        elif arch == "densenet169":
            block_config = (6, 12, 32, 32)
        else:
            raise ValueError("arch must be 'densenet121' or 'densenet169'")

        self.features = DenseNetFeatures(
            in_channels=in_channels,
            growth_rate=32,
            block_config=block_config,
            num_init_features=64,
            bn_size=4,
            compression=0.5,
            drop_rate=drop_rate
        )
        self.CR = CR(8,overlap=1/5)
        self.lif = mem_update()

        feat_ch = self.features.out_channels
        self.classifier_conv = nn.Conv2d(feat_ch, num_classes, kernel_size=1, bias=True)
        
    def update_p_epoch(self, epoch: int, num_epochs: int = 20, p_start: float = 0.1, p_end: float = 1.0):
            """
            在每个 epoch 开头调用一次：
                model.update_p_epoch(epoch)

            - p 会从 p_start 平滑过渡到 p_end(p_start < p_end 就是“增长”，反之就是“下降”）
            - p 会以每个 epoch 增加 0.1，直到 p 达到 1.0
            """
            # 计算更新的 p 值
            p = p_start + 0.1 * epoch  # 每个 epoch 增加 0.1
            p = min(p, p_end)  # 保证 p 最大值为 p_end（1.0）

            # 下发到整网所有 MultiSpike4
            for m in self.modules():
                if isinstance(m, MultiSpike4):
                    m.set_p(p)

            return p   

    def forward(self, x):
        # [B, Cin, T, F] -> [B, Cin, F, T]
        # print(f"x",x.shape)
        x = self.CR(x)
        # print(f"x",x.shape)
        x = x.permute(0, 1, 3, 2).contiguous()

        feat = self.features(x)          # [B, C, F', T']
        # feat = F.relu(feat, inplace=False)
        feat=self.lif(feat)

        feat = feat.mean(dim=2, keepdim=True)  # [B, C, 1, T']
        out = self.classifier_conv(feat)       # [B, num_classes, 1, T']
        out = out.squeeze(2)                   # [B, num_classes, T']
        return out