import modal
from modal import Stub, Image, Volume, Secret, Mount, gpu

from datetime import datetime
import subprocess
import secrets
import shutil
import glob
import os

N_GPUS = os.environ.get("N_GPUS", 2)
GPU_MEM = os.environ.get("GPU_MEM", 80)
GPU_CONFIG = gpu.A100(count=N_GPUS, memory=GPU_MEM)
APP_NAME = "example-axolotl"

image = (
    Image.from_registry("winglian/axolotl:main-py3.10-cu118-2.0.1")
    .run_commands("git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl")
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

stub = Stub(APP_NAME, secrets=[Secret.from_name("huggingface")])

# Download pre-trained models into this volume.
pretrained_volume = Volume.persisted("example-pretrained-vol")

# Save trained models into this volume.
runs_volume = Volume.persisted("example-runs-vol")

VOLUME_CONFIG = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}


@stub.function(image=image, gpu=GPU_CONFIG, volumes=VOLUME_CONFIG, timeout=3600 * 24)
def train(run_id: str):
    folder = f"/runs/{run_id}"

    print(f"Starting training run in {folder}")
    subprocess.call(["accelerate", "launch", "-m", "axolotl.cli.train", "./config.yml"], cwd=folder)
    print("Committing results to", folder)
    runs_volume.commit()

    subprocess.call(["python3", "-m", "axolotl.cli.merge_lora", "./config.yml", "--load_in_8bit=False", "--load_in_4bit=False"], cwd=folder)
    print("Committing merged weights to", folder)
    runs_volume.commit()

AXOLOTL_MOUNTS = [Mount.from_local_dir("./axolotl-mount", remote_path="/root")]

@stub.function(image=image, timeout=60 * 30, mounts=AXOLOTL_MOUNTS, volumes=VOLUME_CONFIG)
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
    # Generate a run_id from the current timestamp
    run_id = f"axo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{secrets.token_hex(2)}"
    os.makedirs(f"/runs/{run_id}")
    shutil.move(config, f"/runs/{run_id}/config.yml")
    for file in glob.glob("*.jsonl"):
        shutil.move(file, f"/runs/{run_id}")
    runs_volume.commit()
    print(f"Prepared training run {run_id}.")

    # Start a training run.
    return train.spawn(run_id)

@stub.function(volumes=VOLUME_CONFIG, timeout=3600)
def browse():
    with modal.forward(8000) as tunnel:
        print("Volume available at", tunnel.url)
        subprocess.call(["python", "-m", "http.server", "8000", "-d", "/runs"])

@stub.function(image=image, gpu=gpu.A100(), volumes=VOLUME_CONFIG, timeout=3600)
def infer(run_id: str):
    lora_folder = f"/runs/{run_id}/lora-out"
    config = f"/runs/{run_id}/config.yml"

    with modal.forward(7860) as tunnel:
        print("Gradio interface available at", tunnel.url)
        subprocess.call(["accelerate", "launch", "-m", "axolotl.cli.inference", config, "--lora_model_dir", lora_folder, "--gradio"])


@stub.local_entrypoint()
def main():
    train_handle = new.remote("config.yml")
    train_handle.get()
