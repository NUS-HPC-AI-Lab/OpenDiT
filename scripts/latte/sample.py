# Adapted from Latte

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte: https://github.com/Vchitect/Latte
# --------------------------------------------------------

import argparse
import os

import colossalai
import imageio
import torch
from colossalai.cluster import DistCoordinator
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer

from opendit.core.pab_mgr import set_pab_manager
from opendit.core.parallel_mgr import set_parallel_manager
from opendit.models.latte import LattePipeline, LatteT2V
from opendit.utils.utils import merge_args, set_seed


def main(args):
    set_seed(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    set_parallel_manager(coordinator.world_size)
    device = f"cuda:{torch.cuda.current_device()}"

    if args.cross_broadcast or args.spatial_broadcast or args.temporal_broadcast:
        set_pab_manager(
            steps=args.num_sampling_steps,
            cross_broadcast=args.cross_broadcast,
            cross_threshold=args.cross_threshold,
            cross_gap=args.cross_gap,
            spatial_broadcast=args.spatial_broadcast,
            spatial_threshold=args.spatial_threshold,
            spatial_gap=args.spatial_gap,
            temporal_broadcast=args.temporal_broadcast,
            temporal_threshold=args.temporal_threshold,
            temporal_gap=args.temporal_gap,
            diffusion_skip=args.diffusion_skip,
            diffusion_skip_timestep=args.diffusion_skip_timestep,
        )

    transformer_model = LatteT2V.from_pretrained(
        args.pretrained_model_path, subfolder="transformer", video_length=args.video_length
    ).to(device, dtype=torch.float16)

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(
            device
        )
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "HeunDiscrete":
        scheduler = HeunDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "KDPM2AncestralDiscrete":
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )

    videogen_pipeline = LattePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device)

    os.makedirs(args.save_img_path, exist_ok=True)

    # video_grids = []
    for num_prompt, prompt in enumerate(args.text_prompt):
        print("Processing the ({}) prompt".format(prompt))
        videos = videogen_pipeline(
            prompt,
            video_length=args.video_length,
            height=args.image_size[0],
            width=args.image_size[1],
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=args.enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
        ).video
        if coordinator.is_master():
            if videos.shape[1] == 1:
                save_image(videos[0][0], args.save_img_path + prompt[:30].replace(" ", "_") + ".png")
            else:
                imageio.mimwrite(
                    args.save_img_path + prompt[:30].replace(" ", "_") + "_%04d" % args.run_time + ".mp4",
                    videos[0],
                    fps=8,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_img_path", type=str, default="./samples/latte/")
    parser.add_argument("--pretrained_model_path", type=str, default="maxin-cn/Latte-1")
    parser.add_argument("--model", type=str, default="LatteT2V")
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--image_size", nargs="+")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="learned_range")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DDIM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--enable_temporal_attentions", action="store_true")
    parser.add_argument("--enable_vae_temporal_decoder", action="store_true")
    parser.add_argument("--text_prompt", nargs="+")

    # pab
    parser.add_argument("--spatial_broadcast", action="store_true", help="Enable spatial attention skip")
    parser.add_argument(
        "--spatial_threshold", type=int, nargs=2, default=[100, 800], help="Spatial attention threshold"
    )
    parser.add_argument("--spatial_gap", type=int, default=2, help="Spatial attention gap")
    parser.add_argument("--temporal_broadcast", action="store_true", help="Enable temporal attention skip")
    parser.add_argument(
        "--temporal_threshold", type=int, nargs=2, default=[100, 800], help="Temporal attention threshold"
    )
    parser.add_argument("--temporal_gap", type=int, default=4, help="Temporal attention gap")
    parser.add_argument("--cross_broadcast", action="store_true", help="Enable cross attention skip")
    parser.add_argument("--cross_threshold", type=int, nargs=2, default=[80, 900], help="Cross attention threshold")
    parser.add_argument("--cross_gap", type=int, default=7, help="Cross attention gap")
    parser.add_argument(
        "--diffusion_skip",
        action="store_true",
    )
    parser.add_argument("--diffusion_skip_timestep", nargs="+")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
