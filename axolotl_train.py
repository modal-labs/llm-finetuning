import modal
from modal import Stub, Image, Volume, Secret, Mount, gpu

import secrets
import shutil
import glob
import os

N_GPUS = os.environ.get("N_GPUS", 1)
GPU_MEM = os.environ.get("GPU_MEM", 40)
GPU_CONFIG = gpu.A100(count=N_GPUS, memory=GPU_MEM)

image = (
    Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.8",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu118")
    .pip_install("packaging")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
        "cd /root/axolotl && git checkout 575a082aae3c38762aa66680d9b4657db8b397c4 && pip install -e '.[flash-attn,deepspeed]'"
    )
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3")
    # HACK(2023/11/14): fixes ImportError for flash_attn_2_cuda.cpython
    .run_commands("cd /root && pip uninstall flash-attn -y && pip install flash-attn==2.3.4")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained"))
    .env(dict(TORCH_DISTRIBUTED_DEBUG="DETAIL"))
    # .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

# new_image = ( #     Image.from_registry("winglian/axolotl:main-py3.10-cu118-2.0.1") )

stub = Stub("example-axolotl", image=image, secrets=[Secret.from_name("huggingface")])

# Download pre-trained models into this volume.
stub.pretrained_volume = Volume.persisted("example-pretrained-vol")

# Save trained models into this volume.
stub.runs_volume = Volume.persisted("example-runs-vol")

VOLUME_CONFIG = {
    "/pretrained": stub.pretrained_volume,
    "/runs": stub.runs_volume,
}

AXOLOTL_MOUNTS = [Mount.from_local_dir("./axolotl-mount", remote_path="/root")]

@stub.function(volumes=VOLUME_CONFIG, timeout=3600)
def merge(run_id: str):
    import subprocess

    lora_folder = f"/runs/{run_id}/lora-out"
    config = f"/runs/{run_id}/config.yml"
    subprocess.call(["python3", "-m", "axolotl.cli.merge_lora", config, "--lora_model_dir", lora_folder, "--load_in_8bit=False", "--load_in_4bit=False"])

    print("Commiting merged weights ...")
    stub.runs_volume.commit()


@stub.function(gpu=GPU_CONFIG, mounts=AXOLOTL_MOUNTS, volumes=VOLUME_CONFIG, timeout=3600 * 24)
def train(run_id: str):
    from accelerate import notebook_launcher
    from axolotl.cli import train

    folder = f"/runs/{run_id}"
    notebook_launcher(train.do_cli, args=[f"{folder}/config.yml"], num_processes=N_GPUS)
    print("Committing results to", folder)
    stub.runs_volume.commit()

    # Spawn a job to merge adapter weights on a CPU container.
    merge.spawn(run_id)


@stub.function(timeout=60 * 30, mounts=AXOLOTL_MOUNTS, volumes=VOLUME_CONFIG)
def new(config: str):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    # Ensure the base model is downloaded
    with open(config, "r") as f:
        model_name = f.readline().split(":")[-1].strip()

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} (no progress bar) ...")
        snapshot_download(model_name)
        move_cache()

        print("Committing /pretrained directory (no progress bar) ...")
        stub.pretrained_volume.commit()

    # Create a subfolder for the training run with config and data.
    run_id = f"axo-{secrets.token_hex(3)}"
    os.makedirs(f"/runs/{run_id}")
    shutil.move(config, f"/runs/{run_id}/config.yml")
    for file in glob.glob("*.jsonl"):
        shutil.move(file, f"/runs/{run_id}")
    stub.runs_volume.commit()

    # Spawn a training run.
    train.spawn(run_id)


@stub.function(gpu=gpu.A100(), volumes=VOLUME_CONFIG, timeout=3600)
def infer(run_id: str):
    import subprocess

    lora_folder = f"/runs/{run_id}/lora-out"
    config = f"/runs/{run_id}/config.yml"
    with modal.forward(7860) as tunnel:
        print("Gradio interface available at", tunnel.url)
        subprocess.call(["accelerate", "launch", "-m", "axolotl.cli.inference", config, "--lora_model_dir", lora_folder, "--gradio"])