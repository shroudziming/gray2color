import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_output_activation(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None or activation == "none":
        return x
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unsupported output_activation: {activation!r}. Use 'sigmoid', 'tanh', or None.")

#conv module
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels,mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            #padding=1保持输入输出尺寸不变，bias=False因为后面有BN层
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),   #BN层提升训练速度
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

#downsampling module
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            #maxpool降采样 stride=2 将H,W减半s
            nn.MaxPool2d(2,stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Up,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        #输入x1和x2的尺寸可能不匹配，进行中心裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # 增加padding操作，padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])  #对x1进行padding，使其尺寸与x2匹配
        
        #拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Sequential):
    def __init__(self,in_channels,out_classes):
        super(OutConv,self).__init__(
            nn.Conv2d(in_channels,out_classes,kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_classes: int = 3,
                 base_channels: int = 64,
                 concat_input: bool = False,
                 output_activation: str | None = "sigmoid"):
        
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.concat_input = concat_input
        self.output_activation = output_activation

        # 编码器部分
        self.in_conv = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # 解码器部分
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        out_in_channels = base_channels + in_channels if concat_input else base_channels
        self.out_conv = OutConv(out_in_channels, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x  # optional: used for final concat with input
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if self.concat_input:
            diffY = x0.size()[2] - x.size()[2]
            diffX = x0.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, x0], dim=1)

        out = self.out_conv(x)
        out = _apply_output_activation(out, self.output_activation)
        return out