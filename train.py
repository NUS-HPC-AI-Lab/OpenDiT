# Modified from Meta DiT: https://github.com/facebookresearch/DiT

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from diffusers.models import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.models.latte import Latte_models
from opendit.models.mmdit import MMDiT_models
from opendit.models.mmdit_latte import MMLatte_models
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.data_utils import prepare_dataloader
from opendit.utils.operation import model_sharding
from opendit.utils.pg_utils import ProcessGroupManager
from opendit.utils.train_utils import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, update_ema
from opendit.utils.video_utils import DatasetFromCSV, get_transforms_image, get_transforms_video
from opendit.vae.wrapper import AutoencoderKLWrapper

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({}, seed=args.global_seed)
    coordinator = DistCoordinator()
    device = get_current_device()

    # ==============================
    # Setup an experiment folder
    # ==============================
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    model_string_name = args.model.replace("/", "-")
    # Create an experiment folder
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-{model_string_name}"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir, exist_ok=True)
        with open(f"{experiment_dir}/config.txt", "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if coordinator.is_master():
        tensorboard_dir = f"{experiment_dir}/tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "ddp":
        plugin = TorchDDPPlugin()
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Process Group
    # ==============================
    sp_size = args.sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    pg_manager = ProcessGroupManager(dp_size, sp_size, dp_axis=0, sp_axis=1)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Create VAE encoder
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    if args.use_video:
        # Wrap the VAE in a wrapper that handles video data
        # We use 2d vae from stableai instead of 3d vqvae from videogpt because it has better results
        vae = AutoencoderKLWrapper(vae)
        # Use 3d patch size that is divisible by the input size
        input_size = (args.num_frames, args.image_size, args.image_size)
        for i in range(3):
            assert input_size[i] % vae.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // vae.patch_size[i] for i in range(3)]
    else:
        input_size = args.image_size // 8

    # Set mixed precision
    if args.mixed_precision == "bf16" and args.plugin != "ddp":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" and args.plugin != "ddp":
        dtype = torch.float16
    elif args.mixed_precision == "fp32" and args.plugin == "ddp":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed precision {args.mixed_precision}")

    # Shared model config for two models
    model_config = {
        "input_size": input_size,
        "num_classes": args.num_classes,
        "enable_layernorm_kernel": args.enable_layernorm_kernel,
        "enable_modulate_kernel": args.enable_modulate_kernel,
        "sequence_parallel_size": args.sequence_parallel_size,
        "sequence_parallel_type": args.sequence_parallel_type,
        "text_encoder": args.text_encoder,
    }

    if 'MM' in args.model:
        model_config = {**model_config, 't5_text_encoder': args.t5_text_encoder}

    # Create DiT model
    if "DiT" in args.model:
        if "VDiT" in args.model:
            assert args.use_video, "VDiT model requires video data"
        else:
            assert not args.use_video, "DiT model requires image data"
        model_class = DiT_models[args.model] if 'MM' not in args.model else MMDiT_models[args.model]
    elif "Latte" in args.model:
        assert args.use_video, "Latte model requires video data"
        model_class = Latte_models[args.model] if 'MM' not in args.model else MMLatte_models[args.model]
    else:
        raise ValueError(f"Unknown model {args.model}")
    model = (
        model_class(
            enable_flashattn=args.enable_flashattn,
            sequence_parallel_group=pg_manager.sp_group,
            dtype=dtype,
            **model_config,
        )
        .to(device)
        .to(dtype)
    )

    model_numel = get_model_numel(model)
    logger.info(f"Model params: {format_numel_str(model_numel)}")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    # Create an EMA of the model for use after training
    ema = model_class(**model_config).to(device)
    ema = ema.to(torch.float32)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # Create diffusion
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")

    # Setup optimizer
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0, adamw_mode=True
    )
    # You can use a lr scheduler if you want
    # Recommend if you continue training from a model
    lr_scheduler = None

    # Prepare models for training
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0, sharded=False)
    # important! This enables embedding dropout for classifier-free guidance
    model.train()
    # EMA model should always be in eval mode
    ema.eval()

    # Setup data:
    if args.use_video:
        dataset = DatasetFromCSV(
            args.data_path,
            transform=get_transforms_video(args.image_size),
            num_frames=args.num_frames,
            frame_interval=args.frame_interval,
        )
    else:
        # master process goes first
        if not coordinator.is_master():
            dist.barrier()
        dataset = CIFAR10(args.data_path, transform=get_transforms_image(args.image_size), download=True)
        if coordinator.is_master():
            dist.barrier()
    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        pg_manager=pg_manager,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(
            booster, model, ema, optimizer, lr_scheduler, args.load, args.sequence_parallel_type
        )
        logger.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    # Only shard ema model when using zero2 plugin
    shard_ema = True if args.plugin == "zero2" else False
    if shard_ema:
        model_sharding(ema)

    num_steps_per_epoch = len(dataloader)

    logger.info(f"Training for {args.epochs} epochs...")
    # if resume training, set the sampler start index to the correct value
    dataloader.sampler.set_start_index(sampler_start_idx)
    for epoch in range(start_epoch, args.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                if args.use_video:
                    batch = next(dataloader_iter)
                    x = batch["video"].to(device)
                    y = batch["text"]
                else:
                    x, y = next(dataloader_iter)
                    x = x.to(device)
                    y = y.to(device)

                # VAE encode
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x)
                    if not args.use_video:
                        x = x.latent_dist.sample().mul_(0.18215)

                # Diffusion
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer, sharded=shard_ema)

                # Log loss values:
                all_reduce_mean(loss)
                global_step = epoch * num_steps_per_epoch + step
                pbar.set_postfix({"loss": loss.item(), "step": step, "global_step": global_step})

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % args.log_every == 0:
                    writer.add_scalar("loss", loss.item(), global_step)

                # Save checkpoint
                if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
                    logger.info(f"Saving checkpoint...")
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        args.batch_size,
                        coordinator,
                        experiment_dir,
                        ema_shape_dict,
                        args.sequence_parallel_type,
                        shard_ema,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {experiment_dir}"
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()) + list(Latte_models.keys()) + list(MMDiT_models.keys()) + list(MMLatte_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--use_video", action="store_true", help="Use video data instead of images")
    parser.add_argument("--plugin", type=str, default="zero2")
    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--load", type=str, default=None, help="Path to a checkpoint dir to load")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--t5_text_encoder", type=str, default="google-t5/t5-small")

    parser.add_argument("--data_path", type=str, default="./datasets", help="Path to the dataset")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")

    parser.add_argument("--enable_modulate_kernel", action="store_true", help="Enable triton modulate kernel")
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")
    parser.add_argument("--sequence_parallel_type", type=str)

    args = parser.parse_args()
    main(args)
