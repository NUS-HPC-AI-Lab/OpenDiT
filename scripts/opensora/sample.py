import argparse
import os
import time

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from omegaconf import OmegaConf
from tqdm import tqdm

from opendit.core.pab_mgr import set_pab_manager
from opendit.core.parallel_mgr import enable_sequence_parallel, set_parallel_manager
from opendit.models.opensora import RFLOW, OpenSoraVAE_V1_2, STDiT3_XL_2, T5Encoder, text_preprocessing
from opendit.models.opensora.datasets import get_image_size, get_num_frames, save_sample
from opendit.models.opensora.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opendit.utils.utils import all_exists, create_logger, merge_args, set_seed, str_to_dtype


def main(args):
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == dtype ==
    dtype = str_to_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    set_parallel_manager(coordinator.world_size)
    enable_sequence_parallelism = enable_sequence_parallel()
    device = f"cuda:{torch.cuda.current_device()}"
    set_seed(seed=args.seed)

    # == init pab ==
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

    # == init logger ==
    logger = create_logger()
    logger.info(f"Inference configuration: {args}\n")
    verbose = args.verbose
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = T5Encoder(
        from_pretrained="DeepFloyd/t5-v1_1-xxl", model_max_length=300, device=device, shardformer=args.enable_t5_speedup
    )
    vae = (
        OpenSoraVAE_V1_2(
            from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
            micro_frame_size=17,
            micro_batch_size=4,
        )
        .to(device, dtype)
        .eval()
    )

    # == prepare video size ==
    image_size = args.image_size
    if image_size is None:
        resolution = args.resolution
        aspect_ratio = args.aspect_ratio
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(args.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = (
        STDiT3_XL_2(
            from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
            qk_norm=True,
            enable_flash_attn=True,
            enable_layernorm_kernel=True,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
        )
        .to(device, dtype)
        .eval()
    )
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build scheduler ==
    scheduler = RFLOW(use_timestep_transform=True, num_sampling_steps=30, cfg_scale=7.0)

    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = args.prompt
    if prompts is None:
        assert args.prompt_path is not None
        prompts = load_prompts(args.prompt_path)

    # == prepare reference ==
    reference_path = args.reference_path if args.reference_path is not None else [""] * len(prompts)
    mask_strategy = args.mask_strategy if args.mask_strategy is not None else [""] * len(prompts)
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = args.fps
    save_fps = fps // args.frame_interval
    multi_resolution = args.multi_resolution
    batch_size = args.batch_size
    num_sample = args.num_sample
    loop = args.loop
    condition_frame_length = args.condition_frame_length
    condition_frame_edit = args.condition_frame_edit
    align = args.align

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    prompt_as_path = args.prompt_as_path

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_idx=idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if args.llm_refine:
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and coordinator.is_master()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=args.aes,
                    flow=args.flow,
                    camera_motion=args.camera_motion,
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                video_clips.append(samples)

            # == save samples ==
            if coordinator.is_master():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    video = [video_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):
                        video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                    )
                    if save_path.endswith(".mp4") and args.watermark:
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
    logger.info("Inference finished.")
    logger.info("Saved samples to %s", save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--config", default=None, type=str, help="path to config yaml")
    parser.add_argument("--seed", default=1024, type=int, help="seed for reproducibility")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--flash-attn", action="store_true", help="enable flash attention")
    parser.add_argument("--enable_t5_speedup", action="store_true", help="enable t5 speedup")
    parser.add_argument("--resolution", default=None, type=str, help="resolution")
    parser.add_argument("--multi-resolution", default=None, type=str, help="multi resolution")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")

    # output
    parser.add_argument("--save-dir", default="./samples/opensora", type=str, help="path to save generated samples")
    parser.add_argument("--num-sample", default=1, type=int, help="number of samples to generate for one prompt")
    parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
    parser.add_argument("--verbose", default=2, type=int, help="verbose level")

    # prompt
    parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
    parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
    parser.add_argument("--llm-refine", action="store_true", help="enable LLM refine")

    # image/video
    parser.add_argument("--num-frames", default=None, type=str, help="number of frames")
    parser.add_argument("--fps", default=24, type=int, help="fps")
    parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
    parser.add_argument("--frame-interval", default=1, type=int, help="frame interval")
    parser.add_argument("--aspect-ratio", default=None, type=str, help="aspect ratio (h:w)")
    parser.add_argument("--watermark", action="store_true", help="watermark video")

    # hyperparameters
    parser.add_argument("--num-sampling-steps", default=30, type=int, help="sampling steps")
    parser.add_argument("--cfg-scale", default=7.0, type=float, help="balance between cond & uncond")

    # reference
    parser.add_argument("--loop", default=1, type=int, help="loop")
    parser.add_argument("--align", default=None, type=int, help="align")
    parser.add_argument("--condition-frame-length", default=5, type=int, help="condition frame length")
    parser.add_argument("--condition-frame-edit", default=0.0, type=float, help="condition frame edit")
    parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
    parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    parser.add_argument("--aes", default=None, type=float, help="aesthetic score")
    parser.add_argument("--flow", default=None, type=float, help="flow score")
    parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")

    # pab
    parser.add_argument("--spatial_broadcast", action="store_true", help="Enable spatial attention skip")
    parser.add_argument(
        "--spatial_threshold", type=int, nargs=2, default=[540, 920], help="Spatial attention threshold"
    )
    parser.add_argument("--spatial_gap", type=int, default=2, help="Spatial attention gap")
    parser.add_argument("--temporal_broadcast", action="store_true", help="Enable temporal attention skip")
    parser.add_argument(
        "--temporal_threshold", type=int, nargs=2, default=[540, 960], help="Temporal attention threshold"
    )
    parser.add_argument("--temporal_gap", type=int, default=4, help="Temporal attention gap")
    parser.add_argument("--cross_broadcast", action="store_true", help="Enable cross attention skip")
    parser.add_argument("--cross_threshold", type=int, nargs=2, default=[540, 960], help="Cross attention threshold")
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
