from datetime import datetime
from pathlib import Path
import secrets
import modal
import os

from .common import (
    app,
    axolotl_image,
    VOLUME_CONFIG,
)

N_GPUS = int(os.environ.get("N_GPUS", 2))
GPU_CONFIG = os.environ.get("GPU_CONFIG", modal.gpu.H100(count=N_GPUS))

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


def run_cmd(cmd: str, run_folder: str):
    import subprocess

    # Ensure volumes contain latest files.
    VOLUME_CONFIG["/pretrained"].reload()
    VOLUME_CONFIG["/runs"].reload()

    # Propagate errors from subprocess.
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)

    # Commit writes to volume.
    VOLUME_CONFIG["/runs"].commit()


@app.function(
    image=axolotl_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=3600 * 24,
    _allow_background_volume_commits=True,
)
def train(run_folder: str, output_dir: str):
    import torch

    print(f"Starting training run in {run_folder}.")
    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

    run_cmd("accelerate launch -m axolotl.cli.train ./config.yml", run_folder)

    # Kick off CPU job to merge the LoRA weights into base model.
    merge_handle = merge.spawn(run_folder, output_dir)
    with open(f"{run_folder}/logs.txt", "a") as f:
        f.write(f"<br>merge: https://modal.com/logs/call/{merge_handle.object_id}\n")
        print(f"Beginning merge {merge_handle.object_id}.")
    return merge_handle

@app.function(image=axolotl_image, 
              volumes=VOLUME_CONFIG, 
              timeout=3600 * 24,
              _allow_background_volume_commits=True)
def preproc_data(run_folder: str):
    print(f"Preprocessing data.")
    run_cmd("python -m axolotl.cli.preprocess ./config.yml", run_folder)


@app.function(image=axolotl_image, volumes=VOLUME_CONFIG, timeout=3600 * 24)
def merge(run_folder: str, output_dir: str):
    import shutil

    output_path = Path(run_folder) / output_dir
    shutil.rmtree(output_path / "merged", ignore_errors=True)

    with open(f"{run_folder}/config.yml") as config:
        print(f"Merge from {output_path}")

    MERGE_CMD = f"accelerate launch -m axolotl.cli.merge_lora ./config.yml --lora_model_dir='{output_dir}'"
    run_cmd(MERGE_CMD, run_folder)

    VOLUME_CONFIG["/runs"].commit()


@app.function(image=axolotl_image, timeout=60 * 30, volumes=VOLUME_CONFIG)
def launch(config_raw: dict, data_raw: str, run_to_resume: str, preproc_only: bool):
    import yaml
    from huggingface_hub import snapshot_download

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
    run_name = f"axo-{time_string}-{secrets.token_hex(2)}" if not run_to_resume else run_to_resume
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    print(f"Preparing training run in {run_folder}.")
    with (
        open(f"{run_folder}/config.yml", "w") as config_file,
        open(f"{run_folder}/{config['datasets'][0]['path']}", "w") as data_file,
    ):
        config_file.write(config_raw)
        data_file.write(data_raw)
    VOLUME_CONFIG["/runs"].commit()

    if preproc_only:
        print("Spawning container for data preprocessing.")
        launch_handle = preproc_data.spawn(run_folder)
    else:
        # Start training run. 
        print("Spawning container for training.")
        launch_handle = train.spawn(run_folder, config["output_dir"])

    with open(f"{run_folder}/logs.txt", "w") as f:
        lbl = 'train' if not preproc_only else 'preproc'
        f.write(f"{lbl}: https://modal.com/logs/call/{launch_handle.object_id}")
    VOLUME_CONFIG["/runs"].commit()

    return run_name, launch_handle


@app.local_entrypoint()
def main(
    config: str,
    data: str,
    merge_lora: bool = True,
    preproc_only: bool = False,
    run_to_resume: str = None,
):
    # Read config and data source files and pass their contents to the remote function.
    with open(config, "r") as cfg, open(data, "r") as dat:
        run_name, launch_handle = launch.remote(cfg.read(), dat.read(), run_to_resume, preproc_only)

    # Write a local reference to the location on the remote volume with the run
    with open(".last_run_name", "w") as f:
        f.write(run_name)

    # Wait for the training run to finish.
    merge_handle = launch_handle.get()
    if merge_lora and not preproc_only:
        merge_handle.get()

    print(f"Training complete. Run tag: {run_name}")
    print(f"To inspect weights, run `modal volume ls example-runs-vol {run_name}`")
    print(
        f"To run sample inference, run `modal run -q src.inference --run-name {run_name}`"
    )
