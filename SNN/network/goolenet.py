import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogLeNetSeq(nn.Module):
    def __init__(self, out_channels=9, out_steps=None, in_channels=1, aux_logits=False, init_weights=False):
        super().__init__()
        self.aux_logits = aux_logits  

        # 1) 输入通道改成 in_channels
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 2) 输出头改成：保留 out_steps（当作“时间步”），把另一个维度压成1
        #    这里假设你的“时间维”在 H（1498 那一维），freq 在 W（1024 那一维）
        self.pool = nn.AdaptiveAvgPool2d((out_steps, 1))   # -> (N,1024,out_steps,1)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Conv2d(1024, out_channels, kernel_size=1)  # -> (N,out_channels,out_steps,1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.maxpool2(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), ceil_mode=True)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), ceil_mode=True)
        # x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        # x = self.maxpool4(x)
        x = F.max_pool2d(x, kernel_size=(1, 2), ceil_mode=True)

        x = self.inception5a(x)
        x = self.inception5b(x)  # (N,1024,H',W')
        # print(f"x",x.shape)

        x = self.pool(x)         # (N,1024,out_steps,1)
        x = self.dropout(x)
        x = self.classifier(x)   # (N,out_channels,out_steps,1)

        x = x.squeeze(-1)        # -> (N,out_channels,out_steps)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 类Inception，有四个分支
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 四个分支连接起来
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# 辅助分类器：类InceptionAux，包括avepool+conv+fc1+fc2
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
                                 # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)  # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)         # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)  # N x 1024
        x = self.fc2(x)  # N x num_classes
        return x

# 类BasicConv2d，包括conv+relu
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cpu")
    KWS = GoogLeNetSeq().to(device)

    # 这里换成你的模型构造方式
    # model = GoogLeNetSeq(out_channels=9, out_steps=20, in_channels=1, aux_logits=False).to(device)
    # 或者：
    # model = ResNet50(out_channels=9, out_steps=20, in_channels=1).to(device)

    model = GoogLeNetSeq(out_channels=9, out_steps=None, in_channels=1, aux_logits=False).to(device)

    x = torch.randn(1, 1, 1498, 1024, device=device)
    y = model(x)
    print("输出形状:", y.shape) 
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
        model_no_cr = GoogLeNetSeq()
        macs, params = profile(model_no_cr, inputs=(dummy_input, ))
        print(f"\nFLOPs: {macs}, 参数量（再统计一遍）: {params}")
    except ImportError:
        print("\n未安装 thop，跳过FLOPs统计。可用 pip install thop 安装。")
    