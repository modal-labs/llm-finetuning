# Fine-tune any LLM in minutes (ft. LLaMA, CodeLlama, Mistral)

### Tired of prompt engineering? You've come to the right place.

This no-frills guide will take you from a dataset to using a fine-tuned LLM model for inference in the matter of minutes. We use all the recommended, start-of-the-art optimizations for fast results:

* *Deepspeed ZeRO-3* to efficiently shard the base model and training state across multiple GPUs [more info](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)
* *Parameter-efficient fine-tuning* via LoRa adapters for faster convergence
* *Gradient checkpointing* to reduce VRAM footprint, fit larger batches and get higher training throughput

The heavy lifting is done by the [`axolotl` framework](https://github.com/OpenAccess-AI-Collective/axolotl).

Using Modal for fine-tuning means you never have to worry about infrastructure headaches like building images and provisioning GPUs. If a training script runs on Modal, it's reproducible and scalable enough to ship to production right away.

### Just one local dependency - a Modal account

1. Create a [Modal](https://modal.com/) account.
2. Install `modal` in your current Python virtual environment (`pip install modal`)
3. Set up a Modal token in your environment (`python3 -m modal setup`)
4. You need to have a [secret](https://modal.com/secrets) named `huggingface` in your workspace. Populate both `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_TOKEN` with the same key from HuggingFace (settings under API tokens).
5. For some LLaMA models, you need to go to the [Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and agree to the terms and conditions for access (granted instantly).

### Code overview

All the logic lies in `train.py`. Three business Modal functions run in the cloud:

* `launch` prepares a new folder in the `/runs` volume with the training config and data for a new training job. It also ensures the base model is downloaded from HuggingFace.
* `train` takes a prepared folder and performs the training job using the config and data.
* `Inference.completion` can spawn a [vLLM](https://modal.com/docs/examples/vllm_inference#fast-inference-with-vllm-mistral-7b) inference container for any pre-trained or fine-tuned model from a previous training job.

The rest of the code are helpers for _calling_ these three functions. There are two main ways to train:

* Use the GUI to familiarize with the system (recommended for new fine-tuners!)
* Use CLI commands (recommended for power users)

### Using the GUI

Deploy the training backend to expose the three business functions (`launch`, `train`, `completion` in `__init__.py`), then run the Gradio GUI.

```bash
modal deploy src
modal run src.gui
```

The `*.modal.tunnel` link from the latter will take you to the Gradio GUI. There will be three tabs: launch training runs, test out trained models and explore the files on the volume.

*What is the difference between `deploy` and `run`?* The former gives you a deployed app which remains ready on the cloud for invocations anywhere, anytime. The latter gives you an ephemeral app which shuts down once your local command exits. This means your training jobs, that run on the deployed app, continue without your laptop being connected; but your GUI does not waste resources when your laptop disconnects.

### Using the CLI

Test out a simple training job with:

```bash
modal run src.train
```

_`--detach` lets the app continue running even if your client disconnects_.

Training depends on two files: `config.yml` and `my_data.jsonl`. The folder is automatically [mounted](https://modal.com/docs/guide/local-data#mounting-directories) to a remote container. When you make local changes to either of these files, they will be reflected in the next training run.

The default configuration fine-tunes CodeLlama Instruct 7B to understand Modal documentation.

To try a model from a previous run, you can use `modal volume ls examples-runs-vol` to choose a folder, and then:

```bash
modal run -q src.inference --run-folder /runs/axo-2023-11-24-17-26-66e8
```