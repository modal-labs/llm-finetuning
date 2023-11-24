import modal
from modal import Stub, Image, Volume, Secret, Mount, gpu

from datetime import datetime
import subprocess
import secrets
import time
import os

N_GPUS = os.environ.get("N_GPUS", 2)
GPU_MEM = os.environ.get("GPU_MEM", 80)
GPU_CONFIG = gpu.A100(count=N_GPUS, memory=GPU_MEM)
APP_NAME = "example-axolotl"

axolotl_image = (
    Image.from_registry("winglian/axolotl:main-py3.10-cu118-2.0.1")
    .run_commands(
        "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl"
    )
    .pip_install("huggingface_hub==0.17.1", "hf-transfer==0.1.3")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

vllm_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3")
    # Pinned to 11/22/23
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@4cea74c73b2e0981aadfefb3a00e8186d065c897"
    )
)

stub = Stub(APP_NAME, secrets=[Secret.from_name("huggingface")])

# Volumes for pre-trained models and training runs.
pretrained_volume = Volume.persisted("example-pretrained-vol")
runs_volume = Volume.persisted("example-runs-vol")
VOLUME_CONFIG = {"/pretrained": pretrained_volume, "/runs": runs_volume}


@stub.cls(
    gpu=gpu.A100(),
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=60,
    container_idle_timeout=120,
)
class Inference:
    def __init__(self, model_path: str) -> None:
        print(model_path)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(model=model_path, gpu_memory_utilization=0.95)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def completion(self, input: str):
        if not input:
            return

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)


@stub.function(
    image=axolotl_image, gpu=GPU_CONFIG, volumes=VOLUME_CONFIG, timeout=3600 * 24
)
def train(run_id: str):
    folder = f"/runs/{run_id}"

    print(f"Starting training run in {folder}")
    subprocess.call(
        ["accelerate", "launch", "-m", "axolotl.cli.train", "./config.yml"], cwd=folder
    )
    subprocess.call(
        [
            "python3",
            "-m",
            "axolotl.cli.merge_lora",
            "./config.yml",
            "--load_in_8bit=False",
            "--load_in_4bit=False",
        ],
        cwd=folder,
    )

    print("Committing merged weights to", folder)
    runs_volume.commit()


@stub.function(image=axolotl_image, timeout=60 * 30, volumes=VOLUME_CONFIG)
def new(config_raw: str, data_raw: str):
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
        snapshot_download(model_name)
        move_cache()

        print("Committing /pretrained directory (no progress bar) ...")
        stub.pretrained_volume.commit()

    # Write config and data into a training subfolder.
    run_id = f"axo-{datetime.now().strftime('%Y-%m-%d-%H-%M')}-{secrets.token_hex(2)}"
    os.makedirs(f"/runs/{run_id}")
    with open(f"/runs/{run_id}/config.yml", "w") as config_file:
        config_file.write(config_raw)
    with open(f"/runs/{run_id}/{config['datasets'][0]['path']}", "w") as data_file:
        data_file.write(data_raw)
    runs_volume.commit()
    print(f"Prepared training run {run_id}.")

    # Start training run.
    return run_id, train.spawn(run_id)


@stub.local_entrypoint()
def main():
    # Read config.yml and my_data.jsonl and pass them to the new function.
    with open("config.yml", "r") as config, open("my_data.jsonl", "r") as data:
        _, train_handle = new.remote(config.read(), data.read())

    # Wait for the training run to finish.
    train_handle.get()
