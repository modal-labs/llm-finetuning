from modal import Stub, Image, Volume, Secret, Mount, gpu
import os

N_GPUS = os.environ.get("N_GPUS", 4)
GPU_MEM = os.environ.get("GPU_MEM", 80)

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
        "cd /root/axolotl && pip install -e '.[flash-attn,deepspeed]'"
    )
    .pip_install("peft @ git+https://github.com/huggingface/peft.git@18773290938fc632c42ac49f462ab34bd1abd3ea")
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3")
    # HACK(2023/11/14): fixes ImportError for flash_attn_2_cuda.cpython
    .run_commands("cd /root && pip uninstall flash-attn -y && pip install flash-attn==2.3.3")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained"))
    # .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

stub = Stub("example-axolotl", image=image, secrets=[Secret.from_name("huggingface")])

# Download pre-trained models into this volume.
stub.pretrained_volume = Volume.persisted("example-pretrained-vol")

# Save trained models into this volume.
stub.results_volume = Volume.persisted("example-results-vol")

VOLUME_CONFIG = {
    "/pretrained": stub.pretrained_volume,
    "/results": stub.results_volume,
}

AXOLOTL_MOUNTS = [Mount.from_local_dir("./axolotl-mount", remote_path="/root")]

@stub.function(timeout=60 * 20, mounts=AXOLOTL_MOUNTS, volumes=VOLUME_CONFIG)
def prepare(config: str):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    # config should be in ./axolotl-mount
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

@stub.function(gpu=gpu.A100(), mounts=AXOLOTL_MOUNTS, volumes=VOLUME_CONFIG, timeout=3600 * 24)
def train(config: str):
    import subprocess

    subprocess.run(["accelerate", "launch", "-m", "axolotl.cli.train", config])

@stub.local_entrypoint()
def cli(config: str):
    prepare.remote(config)
    train.remote(config)
