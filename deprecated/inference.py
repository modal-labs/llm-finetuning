from modal import Image, gpu, method

import subprocess
import os

from common import stub, BASE_MODELS, VOLUME_CONFIG

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.0.3")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("text-generation", "transformers>=4.33.0")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained"))
)


@stub.function(image=tgi_image, volumes=VOLUME_CONFIG, timeout=60 * 20)
def merge(run_id: str, commit: bool = False):
    from text_generation_server.utils.peft import download_and_unload_peft

    os.mkdirs(f"/results/{run_id}/merged")
    subprocess.call(f"cp /results/{run_id}/*.* /results/{run_id}/merged", shell=True)

    print(f"Merging weights for fine-tuned {run_id=}.")
    download_and_unload_peft(f"/results/{run_id}/merged", None, False)

    if commit:
        print("Committing merged model permanently (can take a few minutes).")
        stub.results_volume.commit()


@stub.cls(
    image=tgi_image,
    gpu=gpu.A100(count=1, memory=40),
    allow_concurrent_inputs=100,
    volumes=VOLUME_CONFIG,
)
class Model:
    def __init__(self, base: str = "", run_id: str = ""):
        from text_generation import AsyncClient
        import socket
        import time

        model = f"/results/{run_id}/merged" if run_id else BASE_MODELS[base]

        if run_id and not os.path.isdir(model):
            merge.local(run_id)  # local = run in the same container

        print(f"Loading {model} into GPU ... ")
        launch_cmd = ["text-generation-launcher", "--model-id", model, "--port", "8000"]
        self.launcher = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL)

        self.client = None
        while not self.client and self.launcher.returncode is None:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(1.0)

        assert self.launcher.returncode is None

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.launcher.terminate()

    @method()
    async def generate(self, prompt: str):
        result = await self.client.generate(prompt, max_new_tokens=512)

        return result.generated_text


@stub.local_entrypoint()
def main(prompt: str, base: str, run_id: str = "", batch: int = 1):
    print(f"Running completion for prompt:\n{prompt}")

    print("=" * 20 + "Generating without adapter" + "=" * 20)
    for output in Model(base).generate.map([prompt] * batch):
        print(output)

    if run_id:
        print("=" * 20 + "Generating with adapter" + "=" * 20)
        for output in Model(base, run_id).generate.map([prompt] * batch):
            print(output)
