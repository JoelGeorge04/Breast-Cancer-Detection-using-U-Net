import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution block for lightweight models."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class BreastCancerClassifier(nn.Module):
    """
    CNN classifier for binary breast cancer classification
    Input: RGB images (3 channels)
    Output: Binary classification (0: non-cancerous, 1: cancerous)
    """
    def __init__(self, input_size=128):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Dropout2d(0.25),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Dropout2d(0.25),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Dropout2d(0.25),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Dropout2d(0.25),
        )
        
        # Calculate the flattened size based on input_size
        feature_map_size = input_size // 16
        flattened_size = 256 * feature_map_size * feature_map_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1),  # Binary classification
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simpler CNN classifier for faster training
    """
    def __init__(self, input_size=128):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        feature_map_size = input_size // 16
        flattened_size = 512 * feature_map_size * feature_map_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyCNN(nn.Module):
    """Very small CNN intended for ensemble diversity and fast training."""

    def __init__(self, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MobileStyleCNN(nn.Module):
    """MobileNet-style classifier using depthwise-separable blocks."""

    def __init__(self, input_size=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 384, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(384, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x)


class WideShallowCNN(nn.Module):
    """Wider but shallower architecture to diversify ensemble errors."""

    def __init__(self, input_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


MODEL_REGISTRY = {
    'complex': {'class': BreastCancerClassifier, 'tags': ('baseline',)},
    'simple': {'class': SimpleCNN, 'tags': ('baseline',)},
    'tiny': {'class': TinyCNN, 'tags': ('ensemble', 'lightweight')},
    'mobile': {'class': MobileStyleCNN, 'tags': ('ensemble', 'lightweight')},
    'wide': {'class': WideShallowCNN, 'tags': ('ensemble', 'lightweight')},
}

ENSEMBLE_MODEL_TYPES = tuple(
    model_name for model_name, meta in MODEL_REGISTRY.items() if 'ensemble' in meta['tags']
)


def build_classifier(model_type='complex', input_size=128):
    """Factory for creating classifier models by name."""
    key = model_type.lower().strip()
    if key not in MODEL_REGISTRY:
        allowed = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_type '{model_type}'. Allowed: {allowed}")
    return MODEL_REGISTRY[key]['class'](input_size=input_size)


def get_model_types_by_tag(tag):
    """Return available model keys that include the given tag."""
    tag = tag.lower().strip()
    return [name for name, meta in MODEL_REGISTRY.items() if tag in meta['tags']]


if __name__ == "__main__":
    # Test models
    x = torch.randn(4, 3, 128, 128)  # Batch of 4 images
    print(f"Input shape: {x.shape}")

    for model_name in MODEL_REGISTRY:
        model = build_classifier(model_name, input_size=128)
        output = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{model_name}:")
        print(f"Total parameters: {n_params:,}")
        print(f"Output shape: {output.shape}")

    print(f"\nEnsemble-tagged models: {', '.join(ENSEMBLE_MODEL_TYPES)}")
