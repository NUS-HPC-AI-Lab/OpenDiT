import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def str_to_dtype(x: str):
    if x == "fp32":
        return torch.float32
    elif x == "fp16":
        return torch.float16
    elif x == "bf16":
        return torch.bfloat16
    else:
        raise RuntimeError(f"Only fp32, fp16 and bf16 are supported, but got {x}")


def batch_func(func, *args):
    """
    Apply a function to each element of a batch.
    """
    batch = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.shape[0] == 2:
            batch.append(func(arg))
        else:
            batch.append(arg)

    return batch


def merge_args(args1, args2):
    """
    Merge two argparse Namespace objects.
    """
    if args2 is None:
        return args1

    for k in args2._content.keys():
        if k in args1.__dict__:
            v = getattr(args2, k)
            if isinstance(v, ListConfig) or isinstance(v, DictConfig):
                v = OmegaConf.to_object(v)
            setattr(args1, k, v)
        else:
            raise RuntimeError(f"Unknown argument {k}")

    return args1


def all_exists(paths):
    return all(os.path.exists(path) for path in paths)


def get_logger():
    return logging.getLogger(__name__)


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger
