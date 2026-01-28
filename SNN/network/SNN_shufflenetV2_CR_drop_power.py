import torch
import torch.nn as nn
import torch.nn.functional as F
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
#-------------------------------------------------------------------
#-------------------------------------------------------------------
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
# ---------------- Bottleneck（保持不变） ----------------
def _is_stride_one(s):
    if isinstance(s, int):
        return s == 1
    else:
        return s[0] == 1 and s[1] == 1

def _is_stride_two(s):
    if isinstance(s, int):
        return s == 2
    else:
        return (s[0] == 2) or (s[1] == 2)

#-------------------------------------------------------------------
class ShuffleV2Block(nn.Module):
    """
    改动：
    - stride 允许 int 或 tuple，如 (2,1) 只在高度方向下采样
    - 当不是 stride=1 时，走 proj 分支（与原版 stride=2 一致的逻辑）
    """
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super().__init__()
        assert (isinstance(stride, int) and stride in [1, 2]) or \
               (isinstance(stride, tuple) and stride[0] in [1, 2] and stride[1] in [1, 2])
        self.stride = stride
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.lif = mem_update()

        outputs = oup - inp

        # 主分支
        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            self.lif,
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            self.lif,
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            # nn.ReLU(inplace=True),
            self.lif,
        ]
        self.branch_main = nn.Sequential(*branch_main)

        # 投影分支（非 stride=1 时存在）
        if not _is_stride_one(stride):
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                self.lif,
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                # nn.ReLU(inplace=True),
                self.lif,
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if _is_stride_one(self.stride):
            # 通道打乱：将输入按通道一分为二
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        b, c, h, w = x.data.size()
        assert (c % 4 == 0), "channels must be divisible by 4 for channel shuffle"
        x = x.reshape(b * c // 2, 2, h * w)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, c // 2, h, w)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    """
    适配你的需求：
    - 输入:  (N, 1, 1498, 1024)
    - 输出:  (N, 9, 375)
    - keep_width=True 时，所有下采样只在高度方向进行（stride=(2,1)）
      如需进一步省显存，可把 keep_width=False（即同时在宽度降采样）
    """
    def __init__(self, in_channels=8, n_class=9, out_steps=True,
                 model_size='1.0x', keep_width=True):
        super().__init__()
        # print('model size is ', model_size)
        self.keep_width = keep_width
        self.out_steps = out_steps
        self.n_class = n_class
        self.CR = CR(8,overlap=1/5)
        self.lif = mem_update()

        self.stage_repeats = [4, 8, 4]
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # 第一层：in_channels 改为 1；stride -> (2,1) 只降高
        input_channel = self.stage_out_channels[1]
        # first_stride = ( 1,2) if self.keep_width else 2
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            # nn.ReLU(inplace=True),
            self.lif,
        )

        # maxpool：同理只降高
        # maxpool_stride = (1,2) if self.keep_width else 2s
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stages
        features = []
        for idxstage, numrepeat in enumerate(self.stage_repeats):
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # 每个 stage 的第一个 block 下采样
                    s = (1,2) if self.keep_width else 2
                    features.append(
                        ShuffleV2Block(input_channel, output_channel,
                                       mid_channels=output_channel // 2, ksize=3, stride=s)
                    )
                else:
                    features.append(
                        ShuffleV2Block(input_channel // 2, output_channel,
                                       mid_channels=output_channel // 2, ksize=3, stride=1)
                    )
                input_channel = output_channel
        self.features = nn.Sequential(*features)

        # 最后 1×1 conv
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            # nn.ReLU(inplace=True)
            self.lif,
        )

        # ——改成序列头——
        # 高度自适应到 1，宽度对齐到 out_steps(=375)
        # self.pool = nn.AdaptiveAvgPool2d((out_steps,1))
        # 1×1 conv -> n_class 通道；输出形状 (N, n_class, 1, out_steps)
        self.cls_head = nn.Conv2d(self.stage_out_channels[-1], n_class, kernel_size=1, bias=True)

        self._initialize_weights()
        
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
        # print(x.shape)
        x = self.CR(x)    
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)          # (N, C, H', W')
        # print(x.shape)
        # x = self.pool(x)               # (N, C, 1, out_steps)
        x = F.max_pool2d(x, kernel_size=(1, 32), ceil_mode=True)
        x = self.cls_head(x)           # (N, n_class, 1, out_steps)
        # print(x.shape)
        x = x.squeeze(3)               # (N, n_class, out_steps)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# # ----------------- quick test -----------------
if __name__ == "__main__":
    device=torch.device("cpu")
    KWS = ShuffleNetV2().to(device)
    model = ShuffleNetV2(in_channels=8, n_class=9, out_steps=True, model_size='0.5x', keep_width=True)
    x = torch.randn(1, 1, 1498, 1024)
    y = model(x)
    print("输出形状:", y.shape)  # 期望: torch.Size([1, 9, 20])
    torch.save(KWS.state_dict(), 'test.model')

    # 统计参数量
    total_params = sum(p.numel() for p in KWS.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")

    # 计算模型大致大小（float32）
    model_size_MB = total_params * 4 / 1024 / 1024
    print(f"模型大小约为: {model_size_MB:.2f} MB (float32精度)")

    # 推荐：详细结构和参数统计（需要 pip install torchinfo）
    try:
        from torchinfo import summary
        print("\n=== 模型结构和每层参数量统计 ===")
        summary(KWS, input_size=(1, 1, 1498, 1024))
    except ImportError:
        print("\n未安装 torchinfo，跳过详细结构统计。可用 pip install torchinfo 安装。")
        # 推荐：详细结构和参数统计（需要 pip install torchinfo）
        # # 可选：统计FLOPs（需要 pip install thop）
    try:
        from thop import profile
        device=torch.device("cpu")
        x = torch.randn(1, 1, 1498, 1024,device=device)
        dummy_input=torch.randn(x.shape)
        model_no_cr = ShuffleNetV2()
        macs, params = profile(model_no_cr, inputs=(dummy_input, ))
        print(f"\nFLOPs: {macs}, 参数量（再统计一遍）: {params}")
    except ImportError:
        print("\n未安装 thop，跳过FLOPs统计。可用 pip install thop 安装。")
