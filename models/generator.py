import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        
        self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet_layers = list(self.resnet.children())[:-2]  # Remove the classification head
        self.encoder = nn.Sequential(*self.resnet_layers)  # Use ResNet as encoder

        self.middle = nn.Sequential(
            self.double_conv(512, 1024),
            SelfAttention(1024)  # Add self-attention here
        )
        self.upconv4 = self.up_conv(1024, 512)
        self.decoder4 = self.double_conv(512 + 256, 512)
        self.res_block4 = ResidualBlock(512)

        self.upconv3 = self.up_conv(512, 256)
        self.decoder3 = self.double_conv(256 + 128, 256)
        self.res_block3 = ResidualBlock(256)

        self.upconv2 = self.up_conv(256, 128)
        self.decoder2 = self.double_conv(128 + 64, 128)
        self.res_block2 = ResidualBlock(128)

        self.upconv1 = self.up_conv(128, 64)
        self.decoder1 = self.double_conv(64 + 64, 64)
        self.res_block1 = ResidualBlock(64)

        self.final_upconv = self.up_conv(64, 32)
        self.final_decoder = self.double_conv(32 + 64, 32)
        self.final_res_block = ResidualBlock(32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.upconv_extra = self.up_conv(32, 32)  # Additional upsampling layer
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size.size(2)) // 2
        diff_x = (layer_width - target_size.size(3)) // 2
        return layer[:, :, diff_y:diff_y + target_size.size(2), diff_x:diff_x + target_size.size(3)]
    
    def _resize_to_match(self, source, target):
        return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)
    
    def forward(self, x):
        original_size = x.size()[-2:]  # Save the original size for resizing the output
        # Repeat the single channel input to create a 3-channel input
        x = x.repeat(1, 3, 1, 1)
        
        # Encoding
        e1 = self.encoder[0](x)  # Conv1
        e2 = self.encoder[1](e1)  # BN1
        e3 = self.encoder[2](e2)  # ReLU
        e4 = self.encoder[3](e3)  # MaxPool
        e5 = self.encoder[4](e4)  # Layer1
        e6 = self.encoder[5](e5)  # Layer2
        e7 = self.encoder[6](e6)  # Layer3
        e8 = self.encoder[7](e7)  # Layer4
        
        # Bottleneck
        m = self.middle(e8)
        
        # Decoding
        d4 = self.upconv4(m)
        d4 = self.center_crop(d4, e7)
        d4 = torch.cat([d4, e7], dim=1)
        d4 = self.decoder4(d4)
        d4 = self.res_block4(d4)
        
        d3 = self.upconv3(d4)
        d3 = self.center_crop(d3, e6)
        d3 = torch.cat([d3, e6], dim=1)
        d3 = self.decoder3(d3)
        d3 = self.res_block3(d3)
        
        d2 = self.upconv2(d3)
        d2 = self.center_crop(d2, e5)
        d2 = torch.cat([d2, e5], dim=1)
        d2 = self.decoder2(d2)
        d2 = self.res_block2(d2)
        
        d1 = self.upconv1(d2)
        d1 = self.center_crop(d1, e4)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.decoder1(d1)
        d1 = self.res_block1(d1)
        
        d0 = self.final_upconv(d1)
        d0 = self.center_crop(d0, e3)
        d0 = torch.cat([d0, e3], dim=1)
        d0 = self.final_decoder(d0)
        d0 = self.final_res_block(d0)
        
        d_extra = self.upconv_extra(d0)
        
        out = self.out_conv(d_extra)
        out = self._resize_to_match(out, x)  # Resize to match the original input size
        return out
