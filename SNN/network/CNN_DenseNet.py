import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import numpy as np
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
class DenseNetTF(nn.Module):
    """
    DenseNet for Time-Frequency (STFT/LPS) input.
    输入:  [B, Cin, T, F]  (你的 __get_feature__ 返回通常是这样)
    内部会转换成 Conv2D 习惯的: [B, Cin, F, T]
    输出:  [B, num_classes, T_out]  (可选插值到 target_len)
    """
    def __init__(self, num_classes=9, in_channels=8, arch="densenet121", pretrained=False):
        super().__init__()

        if arch == "densenet121":
            net = tvm.densenet121(weights=(tvm.DenseNet121_Weights.DEFAULT if pretrained else None))
        elif arch == "densenet169":
            net = tvm.densenet169(weights=(tvm.DenseNet169_Weights.DEFAULT if pretrained else None))
        else:
            raise ValueError("arch must be 'densenet121' or 'densenet169'")
        
        self.CR = CR(8,overlap=1/5)

        # DenseNet 的特征提取部分
        self.features = net.features  # nn.Sequential

        # 修改第一层卷积以适配输入通道数（你的 LPS 通常是 1 通道）
        # DenseNet 默认 conv0: in=3, out=64
        self.features.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False
        )

        # DenseNet 最后输出通道数（121: 1024, 169: 1664）
        feat_ch = net.classifier.in_features

        # 沿频率维聚合后，用 1x1 conv 输出每个时间步的类别 logits
        self.classifier_conv = nn.Conv2d(feat_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x,):
        """
        x: [B, Cin, T, F]
        target_len: 如果给了，就把输出时间长度插值到 target_len
        """
        # 转成 [B, Cin, F, T] 作为 2D 图像输入（H=Freq, W=Time）
        # print(f"x",x.shape)
        x = self.CR(x)
        x = x.permute(0, 1, 3, 2).contiguous()
        # print(f"x",x.shape)

        # DenseNet 特征
        feat = self.features(x)  # [B, C_feat, F', T']
        # print(f"feat",feat.shape)

        # DenseNet 通常最后已包含 norm/relu，稳妥起见再激活一次也行
        feat = F.relu(feat, inplace=False)

        # 沿频率维做平均（把频率压成1），保留时间维
        feat = feat.mean(dim=2, keepdim=True)  # [B, C_feat, 1, T']
        # print(f"feat",feat.shape)

        # 1x1 conv -> [B, num_classes, 1, T']
        out = self.classifier_conv(feat)

        # squeeze freq维 -> [B, num_classes, T']
        out = out.squeeze(2)

        return out
    
# ---------------- 简单测试 ----------------
if __name__ == "__main__":
    # 你的场景：单通道输入，输出 (N,9,20)
    device=torch.device("cpu")
    KWS = DenseNetTF().to(device)
    model = DenseNetTF(num_classes=9, in_channels=8, arch="densenet121", pretrained=False )

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
        # # 可选：统计FLOPs（需要 pip install thop）
    try:
        from thop import profile
        device=torch.device("cpu")
        x = torch.randn(1, 1, 1498, 1024,device=device)
        dummy_input=torch.randn(x.shape)
        model_no_cr = DenseNetTF()
        macs, params = profile(model_no_cr, inputs=(dummy_input, ))
        print(f"\nFLOPs: {macs}, 参数量（再统计一遍）: {params}")
    except ImportError:
        print("\n未安装 thop，跳过FLOPs统计。可用 pip install thop 安装。")