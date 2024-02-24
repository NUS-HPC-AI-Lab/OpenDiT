import argparse
import json
import os
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opendit.models.diffusion import create_diffusion
from opendit.models.dit import DiT, DiT_models
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.data_utils import VideoDataset, prepare_dataloader
from opendit.utils.operation import model_sharding
from opendit.utils.pg_utils import ProcessGroupManager
from opendit.utils.train_utils import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, update_ema

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
    colossalai.launch_from_torch({})
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
    # Setup data:
    dataset = VideoDataset(args.data_path)
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

    # Create model
    img_size = dataset[0][0].shape[-1]
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    model: DiT = (
        DiT_models[args.model](
            input_size=img_size,
            num_classes=args.num_classes,
            enable_flashattn=args.enable_flashattn,
            enable_layernorm_kernel=args.enable_layernorm_kernel,
            enable_modulate_kernel=args.enable_modulate_kernel,
            sequence_parallel_size=args.sequence_parallel_size,
            dtype=dtype,
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
    ema: DiT = (
        DiT_models[args.model](
            input_size=img_size,
            num_classes=args.num_classes,
            enable_flashattn=args.enable_flashattn,
            enable_layernorm_kernel=args.enable_layernorm_kernel,
            enable_modulate_kernel=args.enable_modulate_kernel,
            sequence_parallel_size=args.sequence_parallel_size,
            sequence_parallel_group=None,
            dtype=dtype,
        )
        .to(device)
        .to(dtype)
    )
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")

    # Setup optimizer
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(model.parameters(), lr=args.lr, weight_decay=0, adamw_mode=True)
    # You can use a lr scheduler if you want
    lr_scheduler = None

    # Prepare models for training
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0, sharded=False)
    # important! This enables embedding dropout for classifier-free guidance
    model.train()
    # EMA model should always be in eval mode
    ema.eval()

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
        start_epoch, start_step, sampler_start_idx = load(booster, model, ema, optimizer, lr_scheduler, args.load)
        logger.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")
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
                x, y = next(dataloader_iter)
                x = x.to(device)
                y = y.to(device)

                # TODO: deal with 4d x
                x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
                x = x[:, :4, :, :].contiguous()

                # Diffusion
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer)

                # Log loss values:
                all_reduce_mean(loss)
                if coordinator.is_master() and (step + 1) % args.log_every == 0:
                    pbar.set_postfix({"loss": loss.item()})
                    writer.add_scalar("loss", loss.item(), epoch * num_steps_per_epoch + step)

                if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
                    logger.info(f"Saving checkpoint")
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        args.batch_size,
                        coordinator,
                        experiment_dir,
                        ema_shape_dict,
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step + 1} to {experiment_dir}")

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--plugin", type=str, default="zero2", choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"]
    )
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--enable_modulate_kernel", action="store_true", help="Enable triton modulate kernel")
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")
    args = parser.parse_args()
    main(args)
