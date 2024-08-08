# Adapted from Open-Sora-Plan

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# --------------------------------------------------------

import argparse
import math
import os

import colossalai
import imageio
import torch
from colossalai.cluster import DistCoordinator
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
from opendit.models.opensora_plan import LatteT2V, VideoGenPipeline, ae_stride_config, getae_wrapper
from opendit.utils.utils import merge_args, set_seed


def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding, (padding + w) * ncol + padding, c), dtype=torch.uint8)

    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r : start_r + h, start_c : start_c + w] = video[i]

    return video_grid


def main(args):
    set_seed(42)
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

    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir).to(
        device, dtype=torch.float16
    )
    # vae = getae_wrapper(args.ae)(args.ae_path).to(device, dtype=torch.float16)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # Load model:
    transformer_model = LatteT2V.from_pretrained(
        args.model_path, subfolder=args.version, cache_dir=args.cache_dir, torch_dtype=torch.float16
    ).to(device)
    # transformer_model = LatteT2V.from_pretrained(args.model_path, low_cpu_mem_usage=False, device_map=None, torch_dtype=torch.float16).to(device)

    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(
        args.text_encoder_name, cache_dir=args.cache_dir, torch_dtype=torch.float16
    ).to(device)

    if args.force_images:
        ext = "jpg"
    else:
        ext = "mp4"

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == "DDIM":  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == "DDPM":  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler()
    elif args.sample_method == "HeunDiscrete":  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == "KDPM2AncestralDiscrete":  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    videogen_pipeline = VideoGenPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device=device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith("txt"):
        text_prompt = open(args.text_prompt[0], "r").readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    for idx, prompt in enumerate(args.text_prompt):
        print("Processing the ({}) prompt".format(prompt))
        videos = videogen_pipeline(
            prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=not args.force_images,
            num_images_per_prompt=1,
            mask_feature=True,
        ).video
        try:
            if args.force_images:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(
                    videos / 255.0,
                    os.path.join(args.save_img_path, f"{idx}.{ext}"),
                    nrow=1,
                    normalize=True,
                    value_range=(0, 1),
                )  # t c h w

            else:
                imageio.mimwrite(
                    os.path.join(args.save_img_path, f"{idx}.{ext}"), videos[0], fps=args.fps, quality=9
                )  # highest quality is 10, lowest is 0
        except:
            print("Error when saving {}".format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)

    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    if coordinator.is_master():
        if args.force_images:
            save_image(
                video_grids / 255.0,
                os.path.join(
                    args.save_img_path, f"{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}"
                ),
                nrow=math.ceil(math.sqrt(len(video_grids))),
                normalize=True,
                value_range=(0, 1),
            )
        else:
            video_grids = save_video_grid(video_grids)
            imageio.mimwrite(
                os.path.join(
                    args.save_img_path, f"{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}"
                ),
                video_grids,
                fps=args.fps,
                quality=9,
            )

    print("save path {}".format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.0.0")
    parser.add_argument("--version", type=str, default=None, choices=[None, "65x512x512", "221x512x512", "513x512x512"])
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--force_images", action="store_true")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    # fvd
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
    parser.add_argument("--cross_threshold", type=int, nargs=2, default=[100, 850], help="Cross attention threshold")
    parser.add_argument("--cross_gap", type=int, default=6, help="Cross attention gap")
    parser.add_argument(
        "--diffusion_skip",
        action="store_true",
    )
    parser.add_argument("--diffusion_skip_timestep", nargs="+")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
