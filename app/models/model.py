import torch
import torch.nn as nn
import torch.nn.functional as F





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HybridSegResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters=16, dropout_prob=0.2):
        super(HybridSegResNetUNet, self).__init__()

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, init_filters)
        self.encoder2 = ResidualBlock(init_filters, init_filters * 2, stride=2)
        self.encoder3 = ResidualBlock(init_filters * 2, init_filters * 4, stride=2)
        self.encoder4 = ResidualBlock(init_filters * 4, init_filters * 8, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock(init_filters * 8, init_filters * 16, stride=2)

        # Decoder
        self.decoder4 = nn.ConvTranspose3d(init_filters * 16, init_filters * 8, kernel_size=2, stride=2)
        self.decoder3 = nn.ConvTranspose3d(init_filters * 8, init_filters * 4, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose3d(init_filters * 4, init_filters * 2, kernel_size=2, stride=2)
        self.decoder1 = nn.ConvTranspose3d(init_filters * 2, init_filters, kernel_size=2, stride=2)

        # Final convolution
        self.final_conv = nn.Conv3d(init_filters, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        d4 = self.decoder4(b)
        d4 = self._match_size(d4, e4)  # Ensure sizes match
        d4 = d4 + e4
        d4 = F.relu(d4)

        d3 = self.decoder3(d4)
        d3 = self._match_size(d3, e3)  # Ensure sizes match
        d3 = d3 + e3
        d3 = F.relu(d3)

        d2 = self.decoder2(d3)
        d2 = self._match_size(d2, e2)  # Ensure sizes match
        d2 = d2 + e2
        d2 = F.relu(d2)

        d1 = self.decoder1(d2)
        d1 = self._match_size(d1, e1)  # Ensure sizes match
        d1 = d1 + e1
        d1 = F.relu(d1)

        out = self.final_conv(d1)
        return out

    def _match_size(self, tensor, target):
        # Determine the sizes of the tensor and target
        target_size = target.size()
        current_size = tensor.size()
        
        # Calculate necessary padding or cropping
        pad_d = target_size[2] - current_size[2]
        pad_h = target_size[3] - current_size[3]
        pad_w = target_size[4] - current_size[4]
        
        # Apply padding if needed
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_d // 2, pad_d - pad_d // 2]
            tensor = F.pad(tensor, padding, mode='constant', value=0)
        
        # Apply cropping if needed
        if pad_d < 0 or pad_h < 0 or pad_w < 0:
            tensor = tensor[..., :target_size[2], :target_size[3], :target_size[4]]
        
        return tensor

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your trained model here
    model = HybridSegResNetUNet(in_channels=5, out_channels=3).to(device)
    # model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model

