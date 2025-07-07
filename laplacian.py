from typing import Tuple, List
import einops
import torch
import torch.nn.functional as F

class LaplacianPyramidTorch:
    def __init__(self, levels: int = 4, dtype=torch.float32):
        self.levels = levels
        self.pyramid = []
        self.dtype = dtype

    def gaussian_kernel(self, size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        if size % 2 == 0:
            size += 1
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / torch.sum(kernel)

    def downsample(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        device = image.device
        kernel = self.gaussian_kernel(5, 1.0).to(device).to(self.dtype)
        # apply depth-wise convolution
        kernel = einops.repeat(kernel, "h w -> c 1 h w", c=C)
        blurred = F.conv2d(image, kernel, stride=1, groups=C, padding=2)
        return blurred[..., ::2, ::2]

    def upsample(self, image: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        B, C, H, W = image.shape
        return F.interpolate(image, size=(target_shape[0], target_shape[1]), mode='bilinear', align_corners=False)

    def build_gaussian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        pyramid = [image.clone()]
        for _ in range(self.levels - 1):
            pyramid.append(self.downsample(pyramid[-1]))
        return pyramid

    def build_laplacian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        pyramid = self.build_gaussian_pyramid(image)
        laplacian_pyramid = []
        for i in range(len(pyramid) - 1):
            current = pyramid[i]
            next_level = pyramid[i + 1]
            upsampled = self.upsample(next_level, current.shape[2:])
            laplacian = current - upsampled
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(pyramid[-1])
        return laplacian_pyramid

    def reconstruct_from_laplacian(self, laplacian_pyramid: List[torch.Tensor]) -> torch.Tensor:
        reconstructed = laplacian_pyramid[-1].clone()
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            reconstructed = self.upsample(reconstructed, laplacian_pyramid[i].shape[2:]) + laplacian_pyramid[i]
        return reconstructed