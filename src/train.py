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

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_MEM = int(os.environ.get("GPU_MEM", 80))
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)


def print_common_training_issues(config):
    min_train_tokens = (
        config["sequence_len"]
        * config["gradient_accumulation_steps"]
        * config["micro_batch_size"]
        * N_GPUS
    )
    print(
        f"Please ensure there are enough tokens to train a single epoch of {min_train_tokens} tokens (recommended to have 4x)."
    )

    min_eval_samples = config["micro_batch_size"] * N_GPUS
    print(
        f"Please ensure there are enough samples for evaluation ({min_eval_samples})."
    )


@stub.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    _allow_background_volume_commits=True,
    retries=10,
)
def train(run_folder: str):
    print(f"Starting training run in {run_folder}")

    TRAIN_CMD = "accelerate launch -m axolotl.cli.train ./config.yml"
    if exit_code := subprocess.call(TRAIN_CMD.split(), cwd=run_folder):
        exit(exit_code)

    return merge.spawn(run_folder)


@stub.function(image=axolotl_image, volumes=VOLUME_CONFIG, timeout=3600 * 24)
def merge(run_folder: str):
    import glob

    checkpoints = glob.glob(f"./lora-out/checkpoint-*", root_dir=run_folder)
    MERGE_SRC = max(checkpoints, key=lambda path: int(path.split("-")[-1]))
    print(f"Merge from latest checkpoint {MERGE_SRC} in {run_folder}")

    MERGE_CMD = f"accelerate launch -m axolotl.cli.merge_lora ./config.yml --lora_model_dir='{MERGE_SRC}' --load_in_8bit=False --load_in_4bit=False"
    if exit_code := subprocess.call(MERGE_CMD.split(), cwd=run_folder):
        exit(exit_code)

    VOLUME_CONFIG["/runs"].commit()


@stub.function(image=axolotl_image, timeout=60 * 30, volumes=VOLUME_CONFIG)
def launch(config_raw: str, data_raw: str):
    from huggingface_hub import snapshot_download
    import yaml

    # Ensure the base model is downloaded
    # TODO(gongy): test if this works with a path to previous fine-tune
    config = yaml.safe_load(config_raw)
    model_name = config["base_model"]

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        snapshot_download(model_name)

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
    train_handle = train.spawn(run_folder)
    with open(f"{run_folder}/logs.txt", "w") as f:
        f.write(f"Logs at https://modal.com/logs/call/{train_handle.object_id}\n")
    VOLUME_CONFIG["/runs"].commit()

    return run_folder, train_handle


@stub.local_entrypoint()
def main(config: str = "config.yml", dataset: str = "my_data.jsonl"):
    # Read config.yml and my_data.jsonl and pass them to the new function.
    dir = os.path.dirname(__file__)
    with open(f"{dir}/{config}", "r") as cfg, open(f"{dir}/{dataset}", "r") as data:
        _, train_handle = launch.remote(cfg.read(), data.read())

    # Wait for the training run to finish.
    merge_handle = train_handle.get()
    merge_handle.get()
