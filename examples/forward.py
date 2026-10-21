"""Synthetic CPU example. Random weights do not produce trained denoising."""

import torch

from tccrnet import TCCRNet


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = TCCRNet().eval()
    noisy = torch.randn(2, 1, 512)
    with torch.inference_mode():
        denoised, stats = model(noisy)
    print("Input:", tuple(noisy.shape))
    print("Output:", tuple(denoised.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Finite output:", bool(torch.isfinite(denoised).all()))
    print("Local mixture gate:", float(stats["local_mix"]))


if __name__ == "__main__":
    main()
