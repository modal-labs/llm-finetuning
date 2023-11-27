from modal import Stub, Image, Volume, Secret

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
stub.pretrained_volume = Volume.persisted("example-pretrained-vol")
stub.runs_volume = Volume.persisted("example-runs-vol")
VOLUME_CONFIG = {"/pretrained": stub.pretrained_volume, "/runs": stub.runs_volume}
