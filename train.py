from modal import gpu, Mount

from common import stub, N_GPUS, GPU_MEM, BASE_MODELS, VOLUME_CONFIG


@stub.function(
    volumes=VOLUME_CONFIG,
    memory=1024 * 100,
    timeout=3600 * 4,
)
def download(model_name: str):
    from huggingface_hub import snapshot_download, login
    from transformers.utils import move_cache
    import os

    hf_key = os.environ["HUGGINGFACE_TOKEN"]
    login(hf_key)

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} (no progress bar) ...")
        snapshot_download(model_name)
        move_cache()

        print("Committing /pretrained directory (no progress bar) ...")
        stub.pretrained_volume.commit()


def library_entrypoint(config):
    from llama_recipes.finetuning import main

    main(**config)


@stub.function(
    volumes=VOLUME_CONFIG,
    mounts=[
        Mount.from_local_dir("./datasets", remote_path="/root"),
    ],
    gpu=gpu.A100(count=N_GPUS, memory=GPU_MEM),
    timeout=3600 * 12,
)
def train(train_kwargs):
    from torch.distributed.run import elastic_launch, parse_args, config_from_args

    torch_args = parse_args(["--nnodes", "1", "--nproc_per_node", str(N_GPUS), ""])
    print(f"{torch_args=}\n{train_kwargs=}")

    elastic_launch(
        config=config_from_args(torch_args)[0],
        entrypoint=library_entrypoint,
    )(train_kwargs)

    print("Committing results volume (no progress bar) ...")
    stub.results_volume.commit()


@stub.local_entrypoint()  # Runs locally to kick off remote training job.
def main(
    dataset: str,
    base: str = "chat7",
    run_id: str = "",
    num_epochs: int = 10,
    batch_size: int = 16,
):
    print(f"Welcome to Modal Llama fine-tuning.")

    model_name = BASE_MODELS[base]
    print(f"Syncing base model {model_name} to volume.")
    download.remote(model_name)

    if not run_id:
        import secrets

        run_id = f"{base}-{secrets.token_hex(3)}"
    elif not run_id.startswith(base):
        run_id = f"{base}-{run_id}"

    print(f"Beginning run {run_id=}.")
    train.remote(
        {
            "model_name": BASE_MODELS[base],
            "output_dir": f"/results/{run_id}",
            "batch_size_training": batch_size,
            "lr": 3e-4,
            "num_epochs": num_epochs,
            "val_batch_size": 1,
            # --- Dataset options ---
            "dataset": "custom_dataset",
            "custom_dataset.file": dataset,
            "batching_strategy": "packing",
            # --- FSDP options ---
            "enable_fsdp": True,
            "low_cpu_fsdp": True,  # Optimization for FSDP model loading (RAM won't scale with num GPUs)
            "fsdp_config.use_fast_kernels": True,  # Only works when FSDP is on
            "fsdp_config.fsdp_activation_checkpointing": True,  # Activation checkpointing for fsdp
            "pure_bf16": True,
            # --- Required for 70B ---
            "fsdp_config.fsdp_cpu_offload": True,
            "fsdp_peft_cpu_offload_for_save": True,  # Experimental
            # --- PEFT options ---
            "use_peft": True,
            "peft_method": "lora",
            "lora_config.r": 16,
            "lora_config.lora_alpha": 16,
        }
    )

    print(f"Training completed {run_id=}.")
    print(
        f"Test: `modal run inference.py --base {base} --run-id {run_id} --prompt '...'`."
    )
