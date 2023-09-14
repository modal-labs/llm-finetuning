from modal import gpu, Mount

from common import stub, N_GPUS, GPU_MEM, BASE_MODELS

@stub.function(
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
    mounts=[
        Mount.from_local_dir("./helpers", remote_path="/root"),
    ],
    gpu=gpu.A100(count=N_GPUS, memory=GPU_MEM),
    timeout=3600 * 12,
)
def train(model_cli_args: list[str]):
    import subprocess

    torch_cli_args = ["--nnodes", "1", "--nproc_per_node", str(N_GPUS)]
    print(f"{torch_cli_args=} {model_cli_args=}")

    subprocess.run(
        [
            "torchrun",
            *torch_cli_args,
            "-m",
            "llama_recipes.finetuning",
            *model_cli_args,
        ]
    )

    print("Committing results volume (no progress bar) ...")
    stub.results_volume.commit()


@stub.function(
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
    memory=1024 * 100,
    timeout=3600 * 4,
)
def download(model_name: str):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} (no progress bar) ...")
        snapshot_download(model_name)
        move_cache()

        print("Committing /pretrained directory (no progress bar) ...")
        stub.pretrained_volume.commit()


@stub.local_entrypoint() # Runs locally to kick off remote job.
def launch(
    dataset: str,
    base: str = "chat7",
    run_id: str = "",
    num_epochs: int = 3,
    batch_size: int = 16,
):
    print(f"Welcome to Modal Llama fine-tuning.")

    model_name = BASE_MODELS[base]
    print(f"Syncing base model {model_name} to volume.")
    download.remote(model_name)

    if not run_id:
        import secrets

        run_id = f"{base}-{secrets.token_hex(3)}"

    model_cli_args = [
        "--model_name",
        model_name,
        "--output_dir",
        f"/results/{run_id}",
        "--batch_size_training",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        # --- Dataset options ---
        "--dataset",
        "custom_dataset",
        "--custom_dataset.file",
        dataset,
        # --- FSDP options ---
        "--enable_fsdp",
        "--low_cpu_fsdp",  # Optimization for FSDP model loading (RAM won't scale with num GPUs)
        "--fsdp_config.use_fast_kernels",  # Only works when FSDP is on
        "--fsdp_config.fsdp_activation_checkpointing",  # Activation checkpointing for fsdp
        "--pure_bf16",
        # --- PEFT options ---
        "--use_peft",
        "--peft_method",
        "lora",
        "--lora_config.r",
        "16",
        "--lora_config.lora_alpha",
        "16",
    ]

    print(f"Beginning run {run_id=}.")
    train.remote(model_cli_args)
