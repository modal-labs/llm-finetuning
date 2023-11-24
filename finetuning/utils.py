@stub.function(volumes=VOLUME_CONFIG, timeout=3600)
def browse():
    with modal.forward(8000) as tunnel:
        print("Volume available at", tunnel.url)
        subprocess.call(["python", "-m", "http.server", "8000", "-d", "/runs"])


@stub.function(gpu=gpu.A100(), image=axolotl_image, volumes=VOLUME_CONFIG, timeout=3600)
def infer(run_id: str):
    lora_folder = f"/runs/{run_id}/lora-out"
    config = f"/runs/{run_id}/config.yml"

    with modal.forward(7860) as tunnel:
        print("Gradio interface available at", tunnel.url)
        subprocess.call(
            [
                "accelerate",
                "launch",
                "-m",
                "axolotl.cli.inference",
                config,
                "--lora_model_dir",
                lora_folder,
                "--gradio",
            ]
        )
