# Modified from Hugging Face Diffusers: https://github.com/huggingface/diffusers

# Copyright (c) Hugging Face, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys 
from .ckpt_utils import create_logger

logger = create_logger(None)

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


_accelerate_available = importlib.util.find_spec("accelerate") is not None
try:
    _accelerate_version = importlib_metadata.version("accelerate")
    logger.debug(f"Successfully imported accelerate version {_accelerate_version}")
except importlib_metadata.PackageNotFoundError:
    _accelerate_available = False

_huggingface_hub_available = importlib.util.find_spec("huggingface_hub") is not None
try:
    _huggingface_hub_version = importlib_metadata.version("huggingface_hub")
    logger.debug(f"Successfully imported huggingface_hub version {_huggingface_hub_version}")
except importlib_metadata.PackageNotFoundError:
    _accelerate_available = False

def is_accelerate_available():
    return _accelerate_available

def is_huggingface_hub_available():
    return _huggingface_hub_available