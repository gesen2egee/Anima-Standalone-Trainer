

# https://www.crosslabs.org//blog/diffusion-with-offset-noise
def apply_noise_offset(latents, noise, noise_offset, adaptive_noise_scale) -> torch.FloatTensor:
    if noise_offset is None:
        return noise
    if adaptive_noise_scale is not None:
        # latent shape: (batch_size, channels, height, width)
        # abs mean value for each channel
        latent_mean = torch.abs(latents.mean(dim=(2, 3), keepdim=True))

        # multiply adaptive noise scale to the mean value and add it to the noise offset
        noise_offset = noise_offset + adaptive_noise_scale * latent_mean
        noise_offset = torch.clamp(noise_offset, 0.0, None)  # in case of adaptive noise scale is negative

    noise = noise + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return noise


def apply_masked_loss(loss, batch, masked_loss_random_strength: float = None) -> torch.FloatTensor:
    if "conditioning_images" in batch:
        # conditioning image is -1 to 1. we need to convert it to 0 to 1
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)  # use R channel
        mask_image = mask_image / 2 + 0.5
        # print(f"conditioning_image: {mask_image.shape}")
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        # alpha mask is 0 to 1
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1) # add channel dimension
        # print(f"mask_image: {mask_image.shape}, {mask_image.mean()}")
    else:
        return loss

    if loss.ndim == 5 and mask_image.ndim == 4:
        mask_image = mask_image.unsqueeze(2)

    # resize to the same size as the loss
    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")

    if masked_loss_random_strength is not None:
        # Generate random strength between [masked_loss_random_strength, 1.0]
        # shape: (B, 1, 1, 1) or (B, 1, 1, 1, 1)
        r_shape = [loss.shape[0]] + [1] * (mask_image.ndim - 1)
        r = torch.rand(r_shape, device=loss.device, dtype=loss.dtype)
        r = masked_loss_random_strength + (1.0 - masked_loss_random_strength) * r
        mask_image = mask_image + (1.0 - mask_image) * r

    loss = loss * mask_image
    return loss
