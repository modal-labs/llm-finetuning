import os
from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "example-axolotl"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Axolotl image hash corresponding to main-20240522-py3.11-cu121-2.2.2
AXOLOTL_REGISTRY_SHA = (
    "8ec2116dd36ecb9fb23702278ac612f27c1d4309eca86ad0afd3a3fe4a80ad5b"
)

ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"

axolotl_image = (
    modal.Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    # .run_commands(
    #     "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
    #     "cd /root/axolotl && git checkout main",
    # )
    .pip_install(
        "huggingface_hub==0.20.3",
        "hf-transfer==0.1.5",
        "wandb==0.16.3",
        "fastapi==0.110.0",
        "pydantic==2.6.3",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="true",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
)

vllm_image = modal.Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
).pip_install(
    "vllm==0.2.6",
    "torch==2.1.2",
)

app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_dict({"ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")}),
        *([modal.Secret.from_name("wandb")] if ALLOW_WANDB else []),
    ],
)

# Volumes for pre-trained models and training runs.
pretrained_volume = modal.Volume.from_name(
    "example-pretrained-vol", create_if_missing=True
)
runs_volume = modal.Volume.from_name("example-runs-vol", create_if_missing=True)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}


class Colors:
    """ANSI color codes"""

    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m"
