# Modified from Meta DiT: https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import argparse

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.utils.download import find_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    input_size = args.image_size // 8

    dtype = torch.float32
    model = (
        DiT_models[args.model](
            input_size=input_size,
            num_classes=args.num_classes,
            enable_flashattn=False,
            enable_layernorm_kernel=False,
            dtype=dtype,
        )
        .to(device)
        .to(dtype)
    )

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Create sampling noise:
    # Labels to condition the model with (feel free to change):
    if args.num_classes == 1000:
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    else:
        class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n = len(class_labels)
    z = torch.randn(n, 4, input_size, input_size, device=device)
    y = torch.tensor(class_labels, device=device)
    y_null = torch.tensor([0] * n, device=device)
    y = torch.cat([y, y_null], 0)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save and display images:
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=DiT_models.keys(), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    args = parser.parse_args()
    main(args)
