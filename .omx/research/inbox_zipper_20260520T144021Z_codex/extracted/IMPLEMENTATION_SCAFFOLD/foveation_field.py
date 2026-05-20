"""
Pseudocode for generating foveation masks and integrating variable‑resolution
decoding into a video codec.  Foveation allocates more bytes to regions of
interest (e.g. near the vanishing point or around objects) and fewer bytes
elsewhere.  This scaffold focuses on the mask generation and the high‑level
training/export flow.

Key components:

  * `generate_foveation_mask`: produces a per‑pixel weight mask based on a
    vanishing point or saliency map.  Higher weights mean finer resolution.
  * `apply_foveation`: down‑samples or up‑samples regions of the frame
    according to the mask before feeding into the encoder.
  * `train_foveated_decoder`: trains a decoder that takes a foveated input and
    reconstructs the full‑resolution frame.  A custom loss penalises errors
    weighted by the mask.
  * `export_foveated_payload`: serializes the trained model and mask for
    inclusion in the archive.

Actual implementations may use multi‑resolution wavelets, spatially varying
quantization or attention masks.  Here we outline a simple radial mask and
bilinear down‑sampling.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_foveation_mask(shape: Tuple[int, int],
                            center: Tuple[float, float] = None,
                            inner_radius: float = 0.3,
                            outer_radius: float = 0.9) -> np.ndarray:
    """
    Generate a radial foveation mask.

    Pixels near the center have weight 1; weights fall off linearly to 0 at
    the outer radius.  Values outside outer_radius are assigned minimal weight.

    Args:
        shape: (H, W) of the mask.
        center: optional (y, x) center in normalized coordinates [0,1].  If
            None, use the image center.
        inner_radius: radius within which weight=1.
        outer_radius: radius at which weight falls to 0.
    Returns:
        A float32 numpy array of shape (H, W) with values in [0,1].
    """
    H, W = shape
    if center is None:
        cy, cx = 0.5, 0.5
    else:
        cy, cx = center
    y, x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    mask = np.clip((outer_radius - dist) / (outer_radius - inner_radius), 0, 1)
    return mask.astype(np.float32)


def apply_foveation(frame: torch.Tensor, mask: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """
    Apply foveation to a frame by down‑sampling low‑weight regions.

    Args:
        frame: (C, H, W) tensor.
        mask: (H, W) tensor with values in [0,1].
        scale: down‑sample factor for low‑weight pixels.
    Returns:
        Foveated frame tensor of shape (C, H, W).
    """
    C, H, W = frame.shape
    # Create a down‑sampled version
    down = F.interpolate(frame.unsqueeze(0), size=(H // scale, W // scale), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
    # Blend original and upsampled based on mask: high mask values use original; low mask uses blurred
    mask3 = mask.unsqueeze(0).repeat(C, 1, 1)
    return frame * mask3 + up * (1 - mask3)


class FoveatedDecoder(nn.Module):
    """
    Simple decoder that takes a foveated input and reconstructs the full frame.
    This could reuse the HNeRV or VQ‑VAE decoder architecture.
    """
    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(z))
        h = F.relu(self.conv2(h))
        return torch.sigmoid(self.conv3(h))


def train_foveated_decoder(frames: torch.Tensor,
                           masks: torch.Tensor,
                           epochs: int = 100,
                           latent_dim: int = 64,
                           lr: float = 1e-4) -> FoveatedDecoder:
    """
    Train a decoder on foveated inputs.

    Args:
        frames: (T, C, H, W) tensor of original frames in [0,1].
        masks: (T, H, W) tensor of foveation weights.
        epochs: training epochs.
        latent_dim: latent dimensionality (placeholder – in real implementation, encoder produces latents).
        lr: learning rate.
    Returns:
        Trained decoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = FoveatedDecoder(latent_dim=latent_dim, out_channels=frames.size(1)).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    frames = frames.to(device)
    masks = masks.to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Placeholder latent: zeros; replace with encoder outputs in real code
        latent = torch.zeros(frames.size(0), latent_dim, frames.size(2)//4, frames.size(3)//4, device=device)
        recon = decoder(latent)
        # Weighted reconstruction loss
        mask3 = masks.unsqueeze(1).repeat(1, frames.size(1), 1, 1)
        loss = ((recon - frames) ** 2 * mask3).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, foveated loss {loss.item():.6f}")
    return decoder


def export_foveated_payload(model: FoveatedDecoder, masks: torch.Tensor) -> bytes:
    """
    Serialize a trained foveated decoder and its masks.

    Args:
        model: trained FoveatedDecoder.
        masks: tensor of shape (T, H, W) with foveation weights.
    Returns:
        Byte string containing serialized weights and masks.
    """
    import io
    buffer = io.BytesIO()
    torch.save({
        'state_dict': model.state_dict(),
        'masks': masks.cpu().numpy()
    }, buffer)
    return buffer.getvalue()


__all__ = [
    "generate_foveation_mask",
    "apply_foveation",
    "train_foveated_decoder",
    "export_foveated_payload",
]