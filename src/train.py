from datetime import datetime
import subprocess
import secrets
import modal
import os

from .common import (
    stub,
    axolotl_image,
    VOLUME_CONFIG,
)

N_GPUS = os.environ.get("N_GPUS", 2)
GPU_MEM = os.environ.get("GPU_MEM", 80)
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)


@stub.function(
    image=axolotl_image, gpu=GPU_CONFIG, volumes=VOLUME_CONFIG, timeout=3600 * 24
)
def train(run_folder: str):
    print(f"Starting training run in {run_folder}")

    TRAIN_CMD = "accelerate launch -m axolotl.cli.train ./config.yml"
    MERGE_CMD = "accelerate launch -m axolotl.cli.merge_lora ./config.yml --load_in_8bit=False --load_in_4bit=False"

    subprocess.call(TRAIN_CMD.split(), cwd=run_folder)
    subprocess.call(MERGE_CMD.split(), cwd=run_folder)

    print("Committing merged weights to", run_folder)
    VOLUME_CONFIG["/runs"].commit()


# TODO(gongy): rename to start
@stub.function(image=axolotl_image, timeout=60 * 30, volumes=VOLUME_CONFIG)
def launch(config_raw: str, data_raw: str):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    import yaml

    # Ensure the base model is downloaded
    # TODO(gongy): test if this works with a path to previous fine-tune
    config = yaml.safe_load(config_raw)
    model_name = config["base_model"]

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} (no progress bar) ...")
        snapshot_download(model_name, local_dir="/pretrained")

        print("Committing /pretrained directory (no progress bar) ...")
        VOLUME_CONFIG["/pretrained"].commit()

    # Write config and data into a training subfolder.
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_folder = f"/runs/axo-{time_string}-{secrets.token_hex(2)}"
    os.makedirs(run_folder)

    print(f"Preparing training run in {run_folder}.")
    with (
        open(f"{run_folder}/config.yml", "w") as config_file,
        open(f"{run_folder}/{config['datasets'][0]['path']}", "w") as data_file,
    ):
        config_file.write(config_raw)
        data_file.write(data_raw)
    VOLUME_CONFIG["/runs"].commit()

    # Start training run.
    return run_folder, train.spawn(run_folder)


@stub.local_entrypoint()
def main(config: str = "config.yml", dataset: str = "my_data.jsonl"):
    # Read config.yml and my_data.jsonl and pass them to the new function.
    dir = os.path.dirname(__file__)
    with open(f"{dir}/{config}", "r") as cfg, open(f"{dir}/{dataset}", "r") as data:
        _, train_handle = launch.remote(cfg.read(), data.read())

    # Wait for the training run to finish.
    train_handle.get()
