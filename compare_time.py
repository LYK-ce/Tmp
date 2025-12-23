# Presented by KeJi
# Date: 2025-12-22

"""
CNN vs ViT Performance Comparison
Compare inference time of 7M parameter CNN and ViT models on CPU
"""

import os
import torch
import torch.nn as nn
import time
from typing import Tuple

# Limit CPU threads to 1 for fair comparison
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# ================================
# Model Definitions
# ================================

class SimpleCNN(nn.Module):
    """
    7M parameter CNN model based on simplified ResNet architecture
    Target: ~7M parameters
    """
    def __init__(self, num_classes: int = 1000):
        super(SimpleCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = []
        # First block with stride 2 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Basic residual block for SimpleCNN"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out


class SimpleViT(nn.Module):
    """
    7M parameter Vision Transformer
    Target: ~7M parameters
    Config: patch_size=16, embed_dim=192, depth=12, num_heads=3
    """
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: float = 4.0):
        super(SimpleViT, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Layer norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Patch embedding: (B, 3, 224, 224) -> (B, 192, 14, 14) -> (B, 196, 192)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token: (B, 196, 192) -> (B, 197, 192)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification using CLS token
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block for SimpleViT"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


# ================================
# Helper Functions
# ================================

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_model(model: nn.Module, 
                    input_tensor: torch.Tensor, 
                    num_warmup: int = 3,
                    num_runs: int = 10) -> Tuple[float, float, float]:
    """
    Benchmark model inference time
    
    Returns:
        avg_time: Average inference time in milliseconds
        min_time: Minimum inference time in milliseconds
        max_time: Maximum inference time in milliseconds
    """
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return avg_time, min_time, max_time


def main():
    """Main function to compare CNN and ViT performance"""
    print("=" * 60)
    print("CNN vs ViT Performance Comparison on CPU")
    print("=" * 60)
    print()
    
    # Set device
    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print()
    
    # Create models
    print("Creating models...")
    cnn_model = SimpleCNN(num_classes=1000).to(device)
    vit_model = SimpleViT(img_size=224, patch_size=16, embed_dim=384, 
                         depth=16, num_heads=3, num_classes=1000).to(device)
    
    # Count parameters
    cnn_params = count_parameters(cnn_model)
    vit_params = count_parameters(vit_model)
    
    print(f"CNN Model Parameters: {cnn_params:,} ({cnn_params/1e6:.2f}M)")
    print(f"ViT Model Parameters: {vit_params:,} ({vit_params/1e6:.2f}M)")
    print()
    
    # Create input tensor
    batch_size = 1
    img_size = 224
    input_tensor = torch.randn(batch_size, 3, img_size, img_size).to(device)
    print(f"Input shape: {input_tensor.shape}")
    print()
    
    # Benchmark CNN
    print("Benchmarking CNN model...")
    cnn_avg, cnn_min, cnn_max = benchmark_model(cnn_model, input_tensor)
    print(f"  Average: {cnn_avg:.2f} ms")
    print(f"  Min:     {cnn_min:.2f} ms")
    print(f"  Max:     {cnn_max:.2f} ms")
    print()
    
    # Benchmark ViT
    print("Benchmarking ViT model...")
    vit_avg, vit_min, vit_max = benchmark_model(vit_model, input_tensor)
    print(f"  Average: {vit_avg:.2f} ms")
    print(f"  Min:     {vit_min:.2f} ms")
    print(f"  Max:     {vit_max:.2f} ms")
    print()
    
    # Comparison
    print("=" * 60)
    print("Performance Comparison Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Parameters':<15} {'Avg Time':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'CNN':<15} {f'{cnn_params/1e6:.2f}M':<15} {f'{cnn_avg:.2f} ms':>12} {'1.00x':>10}")
    print(f"{'ViT':<15} {f'{vit_params/1e6:.2f}M':<15} {f'{vit_avg:.2f} ms':>12} {f'{cnn_avg/vit_avg:.2f}x':>10}")
    print("=" * 60)
    print()
    
    # Analysis
    if cnn_avg < vit_avg:
        speedup = vit_avg / cnn_avg
        print(f"Result: CNN is {speedup:.2f}x faster than ViT")
        print(f"  CNN: {cnn_avg:.2f} ms")
        print(f"  ViT: {vit_avg:.2f} ms")
        print(f"  Difference: {vit_avg - cnn_avg:.2f} ms ({(speedup-1)*100:.1f}% faster)")
    else:
        speedup = cnn_avg / vit_avg
        print(f"Result: ViT is {speedup:.2f}x faster than CNN")
        print(f"  ViT: {vit_avg:.2f} ms")
        print(f"  CNN: {cnn_avg:.2f} ms")
        print(f"  Difference: {cnn_avg - vit_avg:.2f} ms ({(speedup-1)*100:.1f}% faster)")
    print()
    
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
