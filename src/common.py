from modal import Stub, Image, Volume, Secret
import os

APP_NAME = "example-axolotl"

axolotl_image = (
    Image.from_registry("winglian/axolotl:main-py3.10-cu118-2.0.1")
    .run_commands(
        "git clone https://github.com/OpenAccess-AI-Collective/axolotl /root/axolotl",
        "cd /root/axolotl && git checkout a581e9f8f66e14c22ec914ee792dd4fe073e62f6",
    )
    .pip_install("huggingface_hub==0.19.4", "hf-transfer==0.1.4")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained", HF_HUB_ENABLE_HF_TRANSFER="1"))
)

vllm_image = Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3").pip_install(
    "vllm==0.2.3"
)

stub = Stub(APP_NAME, secrets=[Secret.from_name("huggingface")])

# Volumes for pre-trained models and training runs.
pretrained_volume = Volume.persisted("axo-pretrained-vol")
runs_volume = Volume.persisted("axo-runs-vol")
VOLUME_CONFIG: dict[str | os.PathLike, Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}
