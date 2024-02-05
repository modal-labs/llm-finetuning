# Fine-tune any LLM in minutes (ft. LLaMA, CodeLlama, Mistral)

### Tired of prompt engineering? You've come to the right place.

This no-frills guide will take you from a dataset to using a fine-tuned LLM for inference in the matter of minutes. The heavy lifting is done by the [`axolotl` framework](https://github.com/OpenAccess-AI-Collective/axolotl).

<details>
  <summary>We use all the recommended, state-of-the-art optimizations for fast results.</summary>
  
<br>
  
- *Deepspeed ZeRO-3* to efficiently shard the base model and training state across multiple GPUs ([more info](https://www.deepspeed.ai/2021/03/07/zero3-offload.html))
- *Parameter-efficient fine-tuning* via LoRA adapters for faster convergence
- *Flash attention* for fast and memory-efficient attention during training (note: only works with certain hardware, like A100s)
- *Gradient checkpointing* to reduce VRAM footprint, fit larger batches and get higher training throughput
</details>


Using Modal for fine-tuning means you never have to worry about infrastructure headaches like building images and provisioning GPUs. If a training script runs on Modal, it's reproducible and scalable enough to ship to production right away.

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
modal run --detach src.train --config=config/codellama/config.yml --data=data/sqlqa.jsonl
```

4. Try the model from a completed training run. You can select a folder via `modal volume ls example-runs-vol`, and then specify the training folder with the `--run-folder` flag (something like `/runs/axo-2023-11-24-17-26-66e8`) for inference:

```bash
modal run -q src.inference --run-folder /runs/<run_tag>
```

The default configuration fine-tunes CodeLlama Instruct 7B on a text-to-SQL dataset for five epochs (takes a few minutes) as a proof of concept. It uses DeepSpeed ZeRO-3 to shard the model state across 2 A100s. Inference on the fine-tuned model displays conformity to the output structure (`[SQL] ... [/SQL]`). To achieve better results, you would need to use more data! Refer to the full development section below.

5. (Optional) Launch the GUI for easy observability of training status.

```bash
modal deploy src
modal run src.gui
```

The `*.modal.host` link from the latter will take you to the Gradio GUI. There will be two tabs: (1) launch new training runs, (2) test out trained models.

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

You can `example_configs` for quick start with different models. We recommend duplicating one to `src/config.yml` and modifying as you need. See an overview of Axolotl's config options [here](https://github.com/OpenAccess-AI-Collective/axolotl#config). The most important options to consider are:

**Model**
```yaml
base_model: codellama/CodeLlama-7b-Instruct-hf
```

**Dataset** (by default we upload a local .jsonl file from the `src` folder, but you can see all dataset options [here](https://github.com/OpenAccess-AI-Collective/axolotl#dataset))
```yaml
datasets:
  # This will be the path used for the data when it is saved to the Volume in the cloud.
  - path: my_data.jsonl
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
adapter: lora # for qlora, or leave blank for full finetune
lora_r: 16
lora_alpha: 32 # alpha = 2 x rank is a good rule of thumb.
lora_dropout: 0.05
lora_target_linear: true # target all linear layers
```

### Custom Dataset

Axolotl supports many dataset formats ([see more](https://github.com/OpenAccess-AI-Collective/axolotl#dataset)). We recommend adding your custom dataset as a .jsonl file in the `src` folder and making the appropriate modifications to your config.

**Multi-GPU training**

We recommend [DeepSpeed](https://github.com/microsoft/DeepSpeed) for multi-GPU training, which is easy to set up. Axolotl provides several default deepspeed JSON [configurations](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/deepspeed) and Modal makes it easy to [attach multiple GPUs](https://modal.com/docs/guide/gpu#gpu-acceleration) of any type in code, so all you need to do is specify which of these configs you'd like to use.

In your `config.yml`:
```yaml
deepspeed: /root/axolotl/deepspeed/zero3.json
```

In `train.py`:
```python
N_GPUS = 2
GPU_MEM = 80
GPU_CONFIG = modal.gpu.A100(count=N_GPUS, memory=GPU_MEM) # you can also change this to use A10Gs or T4s
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

The script reads two local files containing the config information and the dataset. The contents passed as arguments to the remote `launch` function, which will write them to the `/runs` volume. Next, `train` will read the config and data from the volume for reproducible training runs.

When you make local changes to either your config or data, they will be used for your next training run.

The default configuration fine-tunes CodeLlama Instruct 7B to understand Modal documentation for five epochs as a proof of concept. It uses DeepSpeed ZeRO-3 to shard the model state across 2 A100s. To achieve better results, you would need to use more data and train for more epochs.

**Inference**

To try a model from a completed run, you can select a folder via `modal volume ls examples-runs-vol`, and then specify the training folder for inference:

```bash
modal run -q src.inference::inference_main --run-folder /runs/axo-2023-11-24-17-26-66e8
```


## Using the GUI

Deploy the training backend with three business functions (`launch`, `train`, `completion` in `__init__.py`). Then run the Gradio GUI.

```bash
modal deploy src
modal run src.gui
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
