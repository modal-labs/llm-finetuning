# Fine-tune any LLM in minutes (ft. Mixtral, LLaMA, Mistral)

This guide will take you from a dataset to using a fine-tuned LLM for inference in a matter of minutes.

## Serverless Axolotl

 This repository gives the popular [`axolotl`](https://github.com/OpenAccess-AI-Collective/axolotl) fine-tuning library a serverless twist. It uses Modal's serverless infrastructure to run your fine-tuning jobs in the cloud, so you can train your models without worrying about building images or idling expensive GPU VMs. 

 Any application written with Modal, including this one, can be trivially scaled across many GPUs in a reproducible and efficient manner.  This makes any fine-tuning job you prototype with this repository production-ready.

### Designed For Efficiency

This tutorial uses many of the recommended, state-of-the-art optimizations for efficient training that axolotl supports, including:
    
- **Deepspeed ZeRO** to utilize multiple GPUs [more info](https://www.deepspeed.ai) during training, according to a strategy you configure.
- **Parameter-efficient fine-tuning** via LoRA adapters for faster convergence
- **Flash attention** for fast and memory-efficient attention during training (note: only works with certain hardware, like A100s)
- **Gradient checkpointing** to reduce VRAM footprint, fit larger batches and get higher training throughput.

### Differences From Axolotl

This modal app does not expose all CLI arguments that axolotl does.  You can specify all your desired options in the config file instead.  However, we find that the interface we provide is sufficient for most users.  Any important differences are noted in the documentation below.

## Quickstart

Follow the steps to quickly train and test your fine-tuned model:
1. Create a Modal account and create a Modal token and HuggingFace secret for your workspace, if not already set up.
    <details> 
    <summary>Setting up Modal</summary>

    1. Create a [Modal](https://modal.com/) account.
    2. Install `modal` in your current Python virtual environment (`pip install modal`)
    3. Set up a Modal token in your environment (`python3 -m modal setup`)
    4. You need to have a [secret](https://modal.com/docs/guide/secrets#secrets) named `huggingface` in your workspace. You can [create a new secret](https://modal.com/secrets) with the HuggingFace template in your Modal dashboard, using the same key from HuggingFace (in settings under API tokens) to populate both `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_TOKEN`.
    5. For some LLaMA models, you need to go to the [Hugging Face page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and agree to their Terms and Conditions for access (granted instantly).
    </details>

2. Clone this repository and navigate to the finetuning directory:
    ```bash
    git clone https://github.com/modal-labs/llm-finetuning.git
    cd llm-finetuning
    ```

3. Launch a training job:
    ```bash
    modal run --detach src.train --config=config/mistral.yml --data=data/sqlqa.jsonl
    ```

Some important caveats about the `train` command:

- The `--data` flag is used to pass your dataset to axolotl. This dataset is then written to the `datasets.path` as specified in your config file. If you alraedy have a dataset at `datasets.path`, you must be careful to also pass the same path to `--data` to ensure the dataset is correctly loaded.
- Unlike axolotl, you cannot pass additional flags to the `train` command. However, you can specify all your desired options in the config file instead.
- This example training script is opinionated in order to make it easy to get started.  For example, a LoRA adapter is used and merged into the base model after training. This merging is currently hardcoded into the `train.py` script.  You will need to modify this script if you do not wish to fine-tune using a LoRA adapter.


4. Try the model from a completed training run. You can select a folder via `modal volume ls example-runs-vol`, and then specify the training folder with the `--run-folder` flag (something like `/runs/axo-2023-11-24-17-26-66e8`) for inference:

```bash
modal run -q src.inference --run-name <run_tag>
```

Our quickstart example trains a 7B model on a text-to-SQL dataset as a proof of concept (it takes just a few minutes). It uses DeepSpeed ZeRO stage 1 to use data parallelism across 2 H100s. Inference on the fine-tuned model displays conformity to the output structure (`[SQL] ... [/SQL]`). To achieve better results, you would need to use more data! Refer to the full development section below.

> [!TIP]
> DeepSpeed ZeRO-1 is not the best choice if your model doesn't comfortably fit on a single GPU. For larger models, we recommend DeepSpeed Zero stage 3 instead by changing the `deepspeed` configuration path.  Modal mounts the [`deepspeed_configs` folder](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/deepspeed_configs) from the `axolotl` repository.  You reference these configurations in your `config.yml` like so: `deepspeed: /root/axolotl/deepspeed_configs/zero3_bf16.json`.  If you need to change these standard configurations, you will need to modify the `train.py` script to load your own custom deepspeed configuration.


5. (Optional) Launch the GUI for easy observability of training status.

```bash
modal deploy src
modal run src.gui
```

The `*.modal.host` link from the latter will take you to the Gradio GUI. There will be two tabs: (1) launch new training runs, (2) test out trained models.

6. Finding your weights

As mentioned earlier, our Modal axolotl trainer automatically merges your LorA adapter weights into the base model weights.  You can browse the artifacts created by your training run with the following command, which is also printed out at the end of your training run in the logs.

```bash
modal volume ls example-runs-vol <run id>
# example: modal volume ls example-runs-vol axo-2024-04-13-19-13-05-0fb0
```

By default, the directory structure will look like this:

```
$ modal volume ls example-runs-vol axo-2024-04-13-19-13-05-0fb0/  

Directory listing of 'axo-2024-04-13-19-13-05-0fb0/' in 'example-runs-vol'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ filename                                       ┃ type ┃ created/modified          ┃ size    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ axo-2024-04-13-19-13-05-0fb0/last_run_prepared │ dir  │ 2024-04-13 12:13:39-07:00 │ 32 B    │
│ axo-2024-04-13-19-13-05-0fb0/mlruns            │ dir  │ 2024-04-13 12:14:19-07:00 │ 7 B     │
│ axo-2024-04-13-19-13-05-0fb0/lora-out          │ dir  │ 2024-04-13 12:20:55-07:00 │ 178 B   │
│ axo-2024-04-13-19-13-05-0fb0/logs.txt          │ file │ 2024-04-13 12:19:52-07:00 │ 133 B   │
│ axo-2024-04-13-19-13-05-0fb0/data.jsonl        │ file │ 2024-04-13 12:13:05-07:00 │ 1.3 MiB │
│ axo-2024-04-13-19-13-05-0fb0/config.yml        │ file │ 2024-04-13 12:13:05-07:00 │ 1.7 KiB │
└────────────────────────────────────────────────┴──────┴───────────────────────────┴─────────┘
```

The LorA adapters are stored in `lora-out`. The merged weights are stored in `lora-out/merged `.   Many inference frameworks can only load the merged weights, so it is handy to know where they are stored.

## Development

### Code overview

All the logic lies in `train.py`. Three business Modal functions run in the cloud:

* `launch` prepares a new folder in the `/runs` volume with the training config and data for a new training job. It also ensures the base model is downloaded from HuggingFace.
* `train` takes a prepared folder and performs the training job using the config and data.
* `Inference.completion` can spawn a [vLLM](https://modal.com/docs/examples/vllm_inference#fast-inference-with-vllm-mistral-7b) inference container for any pre-trained or fine-tuned model from a previous training job.

The rest of the code are helpers for _calling_ these three functions. There are two main ways to train:

* [Use the GUI](#using-the-gui) to familiarize with the system (recommended for new fine-tuners!)
* [Use CLI commands](#using-the-cli) (recommended for power users)

### Config

You can view some example configurations in `config` for a quick start with different models. See an overview of Axolotl's config options [here](https://github.com/OpenAccess-AI-Collective/axolotl#config). The most important options to consider are:

**Model**
```yaml
base_model: mistralai/Mistral-7B-v0.1
```

**Dataset** (You can see all dataset options [here](https://github.com/OpenAccess-AI-Collective/axolotl#dataset))
```yaml
datasets:
  # This will be the path used for the data when it is saved to the Volume in the cloud.
  - path: data.jsonl
    ds_type: json
    type:
      # JSONL file contains question, context, answer fields per line.
      # This gets mapped to instruction, input, output axolotl tags.
      field_instruction: question
      field_input: context
      field_output: answer
      # Format is used by axolotl to generate the prompt.
      format: |-
        [INST] Using the schema context below, generate a SQL query that answers the question.
        {input}
        {instruction} [/INST] 
```

**LoRA**
```yaml
adapter: lora  # for qlora, or leave blank for full finetune (requires much more GPU memory!)
lora_r: 16
lora_alpha: 32  # alpha = 2 x rank is a good rule of thumb.
lora_dropout: 0.05
lora_target_linear: true  # target all linear layers
```

### Custom Dataset

Axolotl supports many dataset formats ([see more](https://github.com/OpenAccess-AI-Collective/axolotl#dataset)). We recommend adding your custom dataset as a .jsonl file in the `data` folder and making the appropriate modifications to your config.

**Multi-GPU training**

We recommend [DeepSpeed](https://github.com/microsoft/DeepSpeed) for multi-GPU training, which is easy to set up. Axolotl provides several default deepspeed JSON [configurations](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/deepspeed) and Modal makes it easy to [attach multiple GPUs](https://modal.com/docs/guide/gpu#gpu-acceleration) of any type in code, so all you need to do is specify which of these configs you'd like to use.

In your `config.yml`:
```yaml
deepspeed: /root/axolotl/deepspeed_configs/zero3_bf16.json
```

In `train.py`:
```python
N_GPUS = 2
GPU_MEM = 40
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM)  # you can also change this to use A10Gs or T4s
```

**Logging with Weights and Biases**

To track your training runs with Weights and Biases:
1. [Create](https://modal.com/secrets/create) a Weights and Biases secret in your Modal dashboard, if not set up already (only the `WANDB_API_KEY` is needed, which you can get if you log into your Weights and Biases account and go to the [Authorize page](https://wandb.ai/authorize))
2. Add the Weights and Biases secret to your app, so initializing your stub in `common.py` should look like: 
```python
stub = Stub(APP_NAME, secrets=[Secret.from_name("huggingface"), Secret.from_name("my-wandb-secret")])
```
3. Add your wandb config to your `config.yml`:
```yaml
wandb_project: code-7b-sql-output
wandb_watch: gradients
```

## Using the CLI

**Training**

A simple training job can be started with

```bash
modal run --detach src.train --config=... --data=...
```

_`--detach` lets the app continue running even if your client disconnects_.

The script reads two local files containing the config information and the dataset. The contents are passed as arguments to the remote `launch` function, which writes them to the `/runs` volume. Finally, `train` reads the config and data from the volume for reproducible training runs.

When you make local changes to either your config or data, they will be used for your next training run.

**Inference**

To try a model from a completed run, you can select a folder via `modal volume ls examples-runs-vol`, and then specify the training folder for inference:

```bash
modal run -q src.inference::inference_main --run-folder=...
```

The training script writes the most recent run name to a local file, `.last_run_name`, for easy access.

## Using the GUI

Deploy the training backend with three business functions (`launch`, `train`, `completion` in `__init__.py`). Then run the Gradio GUI.

```bash
modal deploy src
modal run src.gui --config=... --data=...
```

The `*.modal.host` link from the latter will take you to the Gradio GUI. There will be three tabs: launch training runs, test out trained models and explore the files on the volume.


**What is the difference between `deploy` and `run`?**

- `modal deploy`: a deployed app remains ready on the cloud for invocations anywhere, anytime. This means your training jobs continue without your laptop being connected.
- `modal run`: am ephemeral app shuts down once your local command exits. Your GUI (ephemeral app) does not waste resources when your terminal disconnects.


## Common Errors

> CUDA Out of Memory (OOM)

This means your GPU(s) ran out of memory during training. To resolve, either increase your GPU count/memory capacity with multi-GPU training, or try reducing any of the following in your `config.yml`: micro_batch_size, eval_batch_size, gradient_accumulation_steps, sequence_len

> self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
> ZeroDivisionError: division by zero

This means your training dataset might be too small.

> Missing config option when using `modal run` in the CLI

Make sure your `modal` client >= 0.55.4164 (upgrade to the latest version using `pip install --upgrade modal`)

> AttributeError: 'Accelerator' object has no attribute 'deepspeed_config'

Try removing the `wandb_log_model` option from your config. See [#4143](https://github.com/microsoft/DeepSpeed/issues/4143).
