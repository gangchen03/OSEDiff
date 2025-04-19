import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage

# --- Existing PIL-based functions ---
def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

# --- Helper functions (used by both PIL and Tensor versions) ---
def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    # Calculate var and mean on the feature's device
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    # Ensure calculations happen on the correct device and potentially cast for stability
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (B, C, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor (1, 1, 3, 3)
    kernel = kernel[None, None]
    # repeat the kernel across all input channels (C, 1, 3, 3)
    num_channels = image.shape[1]
    kernel = kernel.repeat(num_channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution with groups=C for depthwise convolution
    output = F.conv2d(image, kernel, groups=num_channels, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    current_low_freq = image
    for i in range(levels):
        radius = 2 ** i
        blurred = wavelet_blur(current_low_freq, radius)
        high_freq += (current_low_freq - blurred)
        current_low_freq = blurred

    return high_freq, current_low_freq # Return the final low frequency component

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, _ = wavelet_decomposition(content_feat)
    # calculate the wavelet decomposition of the style feature
    _, style_low_freq = wavelet_decomposition(style_feat)
    # reconstruct the content feature with the style's low frequency
    # Ensure dtypes match if necessary, though they should be the same
    return content_high_freq + style_low_freq

# --- NEW Tensor-based functions ---

def adain_color_fix_tensor(target_tensor: Tensor, source_tensor: Tensor) -> Tensor:
    """
    Applies AdaIN color correction directly on tensors.
    Args:
        target_tensor (Tensor): The target tensor (e.g., model output) [B, C, H, W].
                                Expected range typically [-1, 1].
        source_tensor (Tensor): The source tensor (e.g., preprocessed input) [B, C, H, W].
                                Expected range typically [-1, 1].
    Returns:
        Tensor: The color-corrected target tensor.
    """
    assert target_tensor.shape == source_tensor.shape, "Target and source tensors must have the same shape."
    # AdaIN works on feature statistics, range [-1, 1] is fine.
    return adaptive_instance_normalization(target_tensor, source_tensor)

def wavelet_color_fix_tensor(target_tensor: Tensor, source_tensor: Tensor) -> Tensor:
    """
    Applies Wavelet color correction directly on tensors.
    Args:
        target_tensor (Tensor): The target tensor (e.g., model output) [B, C, H, W].
                                Expected range typically [-1, 1].
        source_tensor (Tensor): The source tensor (e.g., preprocessed input) [B, C, H, W].
                                Expected range typically [-1, 1].
    Returns:
        Tensor: The color-corrected target tensor.
    """
    assert target_tensor.shape == source_tensor.shape, "Target and source tensors must have the same shape."
    # Wavelet reconstruction works directly on tensor values, range [-1, 1] is fine.
    return wavelet_reconstruction(target_tensor, source_tensor)

