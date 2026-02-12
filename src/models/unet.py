import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convrelu(in_channels, out_channels, kernel, padding):
    """Convolution + ReLU block"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    """U-Net with ResNet18 encoder backbone"""
    
    def __init__(self, n_class):
        super().__init__()
        
        # Load pretrained ResNet18
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())
        
        # Encoder layers
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # Conv1 + BN + ReLU
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # MaxPool + Layer1
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # Layer2
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # Layer3
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # Layer4
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        
        # Decoder layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        # Final output layer
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        logging.info(f"ResNetUNet initialized with {n_class} output class(es)")
    
    def forward(self, input):
        # Original size path
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        # Encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        # Decoder with skip connections
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        
        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        
        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        out = self.conv_last(x)
        
        return out


if __name__ == "__main__":
    """Test model architecture"""
    import torch
    from torchsummary import torchsummary
    
    print("=" * 50)
    print("TESTING ResNetUNet MODEL")
    print("=" * 50)
    
    # Create model
    model = ResNetUNet(n_class=1)
    
    # Test with dummy input
    batch_size = 2
    height, width = 576, 576
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: torch.Size([{batch_size}, 1, {height}, {width}])")
    
    assert output.shape == (batch_size, 1, height, width), "Output shape mismatch!"
    print("\n✓ Forward pass successful")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Test on different input sizes
    print("\nTesting different input sizes:")
    test_sizes = [(256, 256), (512, 512), (640, 640)]
    
    for h, w in test_sizes:
        test_input = torch.randn(1, 3, h, w)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"  Input: {test_input.shape} → Output: {test_output.shape}")
    
    print("\n" + "=" * 50)
    print("✓ ALL MODEL TESTS PASSED")
    print("=" * 50)
