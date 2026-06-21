import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ComplexSteerablePyramid(nn.Module):
    """Complex steerable pyramid decomposition used by CWMI loss."""

    def __init__(self, *, complex: bool = True, levels: int = 4, orientations: int = 4):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if orientations < 1:
            raise ValueError("orientations must be >= 1")

        self.levels = levels
        self.orientations = orientations
        self.complex = complex
        self._mask_cache: Dict[Tuple[int, int, str, torch.dtype], Dict[str, List[torch.Tensor]]] = {}

    @staticmethod
    def _down_sample(fourier_domain_image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = fourier_domain_image.shape
        target_h = max(height // 2, 1)
        target_w = max(width // 2, 1)
        start_h = (height - target_h) // 2
        start_w = (width - target_w) // 2
        end_h = start_h + target_h
        end_w = start_w + target_w
        return fourier_domain_image[:, :, start_h:end_h, start_w:end_w]

    @staticmethod
    def _get_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
        fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype) * (2.0 * math.pi)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx).remainder(2.0 * math.pi)
        radius = torch.fft.fftshift(radius)
        theta = torch.fft.fftshift(theta)
        return radius, theta

    def _get_masks(self, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Dict[str, List[torch.Tensor]]:
        cache_key = (height, width, str(device), dtype)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        high_pass_filters: List[torch.Tensor] = []
        low_pass_filters: List[torch.Tensor] = []
        band_filters: List[torch.Tensor] = []

        current_h = height
        current_w = width
        for level in range(self.levels + 1):
            radius, theta = self._get_grid(current_h, current_w, device=device, dtype=dtype)
            if level == 0:
                high_filter = torch.zeros_like(radius)
                transition = (radius > math.pi / 2.0) & (radius < math.pi)
                high_filter[transition] = torch.cos(
                    (math.pi / 2.0) * torch.log2(radius[transition] / math.pi)
                )
                high_filter[radius >= math.pi] = 1.0

                low_filter = torch.zeros_like(radius)
                low_filter[transition] = torch.cos(
                    (math.pi / 2.0) * torch.log2((2.0 * radius[transition]) / math.pi)
                )
                low_filter[radius <= math.pi / 2.0] = 1.0

                high_pass_filters.append(high_filter)
                low_pass_filters.append(low_filter)
                band_filters.append(torch.empty(0, device=device, dtype=dtype))
            else:
                high_filter = torch.zeros_like(radius)
                transition = (radius > math.pi / 4.0) & (radius < math.pi / 2.0)
                high_filter[transition] = torch.cos(
                    (math.pi / 2.0) * torch.log2((2.0 * radius[transition]) / math.pi)
                )
                high_filter[radius >= math.pi / 2.0] = 1.0

                low_filter = torch.zeros_like(radius)
                low_filter[transition] = torch.cos(
                    (math.pi / 2.0) * torch.log2((4.0 * radius[transition]) / math.pi)
                )
                low_filter[radius <= math.pi / 4.0] = 1.0

                alpha_k = (
                    (2 ** (self.orientations - 1)) * math.factorial(self.orientations - 1)
                ) / math.sqrt(self.orientations * math.factorial(2 * (self.orientations - 1)))
                band = torch.zeros((self.orientations, current_h, current_w), device=device, dtype=dtype)
                for k in range(self.orientations):
                    angle = theta - (math.pi * k / self.orientations)
                    cos_term = torch.cos(angle)
                    if self.complex:
                        directional = 2.0 * torch.abs(alpha_k * torch.relu(cos_term) ** (self.orientations - 1))
                    else:
                        directional = torch.abs(alpha_k * cos_term ** (self.orientations - 1))
                    band[k] = directional
                    band[k, current_h // 2, current_w // 2] = 0.0

                high_pass_filters.append(high_filter)
                low_pass_filters.append(low_filter)
                band_filters.append(band)

                current_h = max(current_h // 2, 1)
                current_w = max(current_w // 2, 1)

        masks = {"high": high_pass_filters, "low": low_pass_filters, "band": band_filters}
        self._mask_cache[cache_key] = masks
        return masks

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape={tuple(images.shape)}")

        _, _, height, width = images.shape
        masks = self._get_masks(height, width, device=images.device, dtype=images.dtype)

        fourier_domain = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
        output: List[torch.Tensor] = []

        high_residue = fourier_domain * masks["high"][0].view(1, 1, height, width)
        output.append(torch.fft.ifft2(torch.fft.ifftshift(high_residue, dim=(-2, -1))))

        current = fourier_domain * masks["low"][0].view(1, 1, height, width)
        for level in range(1, self.levels + 1):
            _, _, h_level, w_level = current.shape
            high_mask = masks["high"][level].view(1, 1, 1, h_level, w_level)
            band_mask = masks["band"][level].view(1, 1, self.orientations, h_level, w_level)
            band_signal = current.unsqueeze(2) * high_mask * band_mask
            output.append(torch.fft.ifft2(torch.fft.ifftshift(band_signal, dim=(-2, -1))))

            low_mask = masks["low"][level].view(1, 1, h_level, w_level)
            current = self._down_sample(current * low_mask)

        output.append(torch.fft.ifft2(torch.fft.ifftshift(current, dim=(-2, -1))))
        if self.complex:
            return output
        return [o.real for o in output]


class CWMILoss(nn.Module):
    """Complex Wavelet Mutual Information loss (full complex formulation)."""

    def __init__(self, *, levels: int = 4, orientations: int = 4, eps: float = 5e-4):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be >= 1")
        if orientations < 1:
            raise ValueError("orientations must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.levels = levels
        self.orientations = orientations
        self.eps = eps
        self._pyramid_cache: Dict[int, ComplexSteerablePyramid] = {}

    @staticmethod
    def _max_supported_levels(height: int, width: int) -> int:
        min_hw = min(height, width)
        if min_hw < 2:
            return 0
        return int(math.floor(math.log2(min_hw)))

    def _get_pyramid(self, levels: int) -> ComplexSteerablePyramid:
        pyramid = self._pyramid_cache.get(levels)
        if pyramid is None:
            pyramid = ComplexSteerablePyramid(complex=True, levels=levels, orientations=self.orientations)
            self._pyramid_cache[levels] = pyramid
        return pyramid

    @staticmethod
    def _to_real_representation(x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, K, HW] complex -> [B, C, 2K, 2HW] real representation
        real = x.real
        imag = x.imag
        upper = torch.cat([real, imag], dim=2)
        lower = torch.cat([-imag, real], dim=2)
        return torch.cat([upper, lower], dim=3)

    def _safe_cholesky(self, matrix: torch.Tensor) -> torch.Tensor:
        dim = matrix.shape[-1]
        identity = torch.eye(dim, device=matrix.device, dtype=matrix.dtype).view(
            *([1] * (matrix.ndim - 2)), dim, dim
        )
        jitter = self.eps
        for _ in range(7):
            chol, info = torch.linalg.cholesky_ex(matrix + identity * jitter)
            if torch.all(info == 0):
                return chol
            jitter *= 10.0

        # Let the final call raise if still not positive definite.
        return torch.linalg.cholesky(matrix + identity * jitter)

    def _complex_negative_mi(self, target_band: torch.Tensor, pred_band: torch.Tensor) -> torch.Tensor:
        # target_band/pred_band: [B, C, K, H, W] complex
        batch_size, channels, _, height, width = target_band.shape
        target_flat = target_band.reshape(batch_size, channels, self.orientations, height * width)
        pred_flat = pred_band.reshape(batch_size, channels, self.orientations, height * width)

        target_real = self._to_real_representation(target_flat)
        pred_real = self._to_real_representation(pred_flat)

        target_centered = target_real - target_real.mean(dim=-1, keepdim=True)
        pred_centered = pred_real - pred_real.mean(dim=-1, keepdim=True)

        n = target_real.shape[-1]
        norm_factor = max(n - 1, 1)

        cov_target = torch.matmul(target_centered, target_centered.transpose(-1, -2)) / norm_factor
        cov_pred = torch.matmul(pred_centered, pred_centered.transpose(-1, -2)) / norm_factor
        cov_target_pred = torch.matmul(target_centered, pred_centered.transpose(-1, -2)) / norm_factor

        dim = cov_pred.shape[-1]
        identity = torch.eye(dim, device=cov_pred.device, dtype=cov_pred.dtype).view(
            *([1] * (cov_pred.ndim - 2)), dim, dim
        )
        inv_cov_pred = torch.linalg.inv(cov_pred + identity * self.eps)

        cond_cov = cov_target - torch.matmul(
            torch.matmul(cov_target_pred, inv_cov_pred), cov_target_pred.transpose(-1, -2)
        )
        cond_cov = 0.5 * (cond_cov + cond_cov.transpose(-1, -2))

        chol = self._safe_cholesky(cond_cov)
        diag = torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1e-12)
        log_det = 2.0 * torch.sum(torch.log(diag), dim=-1)  # [B, C]

        # Eq.7: I ~= -0.5 log det(M). We optimize -I = 0.5 log det(M).
        negative_mi = 0.5 * log_det
        return negative_mi.mean(dim=1)  # [B]

    def forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if target.ndim != 4 or pred.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B, C, H, W], got target={target.ndim}D, pred={pred.ndim}D")
        if target.shape != pred.shape:
            raise ValueError(f"target and pred must have the same shape, got {tuple(target.shape)} vs {tuple(pred.shape)}")

        batch_size, _, height, width = target.shape
        effective_levels = min(self.levels, self._max_supported_levels(height, width))
        if effective_levels <= 0:
            return torch.zeros(batch_size, device=target.device, dtype=target.dtype)

        target_fp32 = target.to(torch.float32)
        pred_fp32 = pred.to(torch.float32)

        pyramid = self._get_pyramid(effective_levels)
        target_bands = pyramid(target_fp32)
        pred_bands = pyramid(pred_fp32)

        loss = torch.zeros(batch_size, device=target.device, dtype=torch.float32)
        for level in range(effective_levels):
            loss = loss + self._complex_negative_mi(target_bands[level + 1], pred_bands[level + 1])
        return loss.to(target.dtype)
