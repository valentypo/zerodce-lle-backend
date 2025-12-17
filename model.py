"""
Zero-DCE (Zero-Reference Deep Curve Estimation) Model Architecture
Extracted from testing_zero_dce.ipynb for backend inference
"""

import torch
import torch.nn as nn


class DCENet(nn.Module):
    """
    Zero-DCE Network Architecture for real-time low-light image enhancement.
    
    The model learns to estimate pixel-wise curve parameters that brighten
    low-light images through iterative enhancement:
    
    Enhanced = Low + α * Low * (1 - Low)
    
    This formula efficiently brightens images while preserving details.
    
    Args:
        n_filters (int): Number of filters in convolutional layers. Default: 32.
        n_iterations (int): Number of enhancement iterations. Default: 8.
    """
    
    def __init__(self, n_filters: int = 32, n_iterations: int = 8):
        super(DCENet, self).__init__()
        self.n_iterations = n_iterations
        
        # Sequential convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        
        # Output layer: generates curve parameters for all iterations
        # Output channels = 3 channels * n_iterations (RGB adjustment for each iteration)
        self.conv7 = nn.Conv2d(n_filters, 3 * n_iterations, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features and apply iterative enhancement.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) with values in [0, 1]
        
        Returns:
            Enhanced tensor of shape (B, 3, H, W) with values in [0, 1]
        """
        # Feature extraction through sequential ReLU convolutions
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))
        x6 = self.relu(self.conv6(x5))
        
        # Generate curve parameters (-1 to 1 range via tanh)
        curve_params = self.tanh(self.conv7(x6))
        
        # Iterative enhancement: apply curve parameters sequentially
        # Formula: I' = I + α * I * (1 - I)
        # This brightens the image for positive α values
        enhanced = x
        for i in range(self.n_iterations):
            alpha = curve_params[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)
        
        # Ensure output is in valid range [0, 1]
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced


def load_model(model_path: str, device: str = 'cpu') -> DCENet:
    """
    Load pre-trained DCENet model from checkpoint.
    
    Args:
        model_path: Path to the .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded DCENet model in evaluation mode
    
    Raises:
        FileNotFoundError: If model file not found
        RuntimeError: If checkpoint format is invalid
    """
    model = DCENet()
    device_obj = torch.device(device)
    
    if not torch.cuda.is_available() and device == 'cuda':
        device_obj = torch.device('cpu')
        print(f"CUDA not available, falling back to CPU")
    
    try:
        # Load checkpoint with weights_only=False since this is trusted model file
        checkpoint = torch.load(model_path, map_location=device_obj, weights_only=False)
        
        # Handle both full checkpoint dict and bare state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from checkpoint (trained for {checkpoint.get('epoch', '?')} epochs)")
        else:
            model.load_state_dict(checkpoint)
            print(f"✓ Loaded model state dictionary")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
    
    model.to(device_obj)
    model.eval()
    
    return model
