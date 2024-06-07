# Fine-tune any LLM in minutes (ft. Mixtral, LLaMA, Mistral)

This guide will show you how to fine-tune any LLM quickly using [`modal`](https://github.com/modal-labs/modal-client) and [`axolotl`](https://github.com/OpenAccess-AI-Collective/axolotl).

## Serverless `axolotl`

Modal gives the popular `axolotl` LLM fine-tuning library serverless superpowers.
If you run your fine-tuning jobs on Modal's cloud infrastructure, you get to train your models without worrying about juggling Docker images or letting expensive GPU VMs sit idle.

And any application written with Modal can be easily scaled across many GPUs --
whether that's several H100 servers running fine-tunes in parallel
or hundreds of A100 or A10G instances running production inference.

### Designed for Efficiency and Performance

Our sample configurations use many of the recommended, state-of-the-art optimizations for efficient, performant training that `axolotl` supports, including:

- [**Deepspeed ZeRO**](https://deepspeed.ai) to utilize multiple GPUs during training, according to a strategy you configure.
- [**LoRA Adapters**]() for fast, parameter-efficient fine-tuning.
- [**Flash attention**](https://github.com/Dao-AILab/flash-attention) for fast and memory-efficient attention calculations during training.

## Quickstart

Our quickstart example overfits a 7B model on a very small subsample of a text-to-SQL dataset as a proof of concept.
Overfitting is a [great way to test training setups](https://fullstackdeeplearning.com/course/2022/lecture-3-troubleshooting-and-testing/#23-use-memorization-testing-on-training)
because it can be done quickly (under five minutes!) and with minimal data but closely resembles the actual training process.

It uses [DeepSpeed ZeRO-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html) to shard model and optimizer state across 2 A100s.

Inference on the fine-tuned model displays conformity to the output structure (`[SQL] ... [/SQL]`). To achieve better results, you'll need to use more data! Refer to the Development section below.

1. **Set up** authentication to Modal for infrastructure, Hugging Face for models, and (optionally) Weights & Biases for training observability:
   <details>
   <summary>Setting up</summary>

   1. Create a [Modal](https://modal.com/) account.
   2. Install `modal` in your current Python virtual environment (`pip install modal`)
   3. Set up a Modal token in your environment (`python3 -m modal setup`)
   4. You need to have a [secret](https://modal.com/docs/guide/secrets#secrets) named `huggingface` in your workspace. You can [create a new secret](https://modal.com/secrets) with the HuggingFace template in your Modal dashboard, using the key from HuggingFace (in settings under API tokens) to populate `HF_TOKEN` and changing the name from `my-huggingface-secret` to `huggingface`.
   5. For some LLaMA models, you need to go to the Hugging Face page (e.g. [this page for LLaMA 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)\_ and agree to their Terms and Conditions for access (granted instantly).
   6. If you want to use [Weights & Biases](https://wandb.ai) for logging, you need to have a secret named `wandb` in your workspace as well. You can also create it [from a template](https://modal.com/secrets). Training is hard enough without good logs, so we recommend you try it or look into `axolotl`'s integration with [MLFlow](https://mlflow.org/)!
   </details>

2. **Clone this repository**:

   ```bash
   git clone https://github.com/modal-labs/llm-finetuning.git
   cd llm-finetuning
   ```

3. **Launch a finetuning job**:
   ```bash
   export ALLOW_WANDB=true  # if you're using Weights & Biases
   modal run --detach src.train --config=config/mistral-memorize.yml --data=data/sqlqa.subsample.jsonl
   ```

This example training script is opinionated in order to make it easy to get started. Feel free to adapt it to suit your needs.

4. **Run inference** for the model you just trained:

```bash
# run one test inference
modal run -q src.inference --prompt "[INST] Using the schema context below, generate a SQL query that answers the question.
CREATE TABLE head (name VARCHAR, born_state VARCHAR, age VARCHAR)
List the name, born state and age of the heads of departments ordered by name.[/INST]"
# 🤖:  [SQL] SELECT name, born_state, age FROM head ORDER BY name [/SQL]
# 🧠: Effective throughput of 36.27 tok/s
```

```bash
# deploy a serverless inference service
modal deploy src.inference
curl https://YOUR_MODAL_USERNAME--example-axolotl-inference-web.modal.run?input=%5BINST%5Dsay%20hello%20in%20SQL%5B%2FINST%5D
# [SQL] Select 'Hello' [/SQL]
```

## Inspecting Flattened Data

One of the key features of axolotl is that it flattens your data from a JSONL file into a prompt template format you specify in the config.
Tokenization and prompt templating are [where most mistakes are made when fine-tuning](https://hamel.dev/notes/llm/05_tokenizer_gotchas.html).

See the [nbs/inspect_data.ipynb](nbs/inspect_data.ipynb) notebook for guide on how to inspect your data and ensure it is being flattened correctly.
We strongly recommend that you always inspect your data the first time you fine-tune a model on a new dataset.

## Development

### Differences from `axolotl`

This Modal app does not expose all configuration via the CLI, the way that axolotl does. You specify all your desired options in the config file instead.

### Code overview

The fine-tuning logic is in `train.py`. These are the important functions:

- `launch` prepares a new folder in the `/runs` volume with the training config and data for a new training job. It also ensures the base model is downloaded from HuggingFace.
- `train` takes a prepared folder and performs the training job using the config and data.
  Some notes about the `train` command:

- The `--data` flag is used to pass your dataset to axolotl. This dataset is then written to the `datasets.path` as specified in your config file. If you already have a dataset at `datasets.path`, you must be careful to also pass the same path to `--data` to ensure the dataset is correctly loaded.
- Unlike `axolotl`, you cannot pass additional flags to the `train` command. However, you can specify all your desired options in the config file instead.
- `--no-merge-lora` will prevent the LoRA adapter weights from being merged into the base model weights.

The `inference.py` file includes a [vLLM](https://modal.com/docs/examples/vllm_inference#fast-inference-with-vllm-mistral-7b) inference server for any pre-trained or fine-tuned model from a previous training job.

### Configuration

You can view some example configurations in `config` for a quick start with different models. See an overview of `axolotl`'s config options [here](https://github.com/OpenAccess-AI-Collective/axolotl#config).

The most important options to consider are:

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
adapter: lora # for qlora, or leave blank for full finetune (requires much more GPU memory!)
lora_r: 16
lora_alpha: 32 # alpha = 2 x rank is a good rule of thumb.
lora_dropout: 0.05
lora_target_linear: true # target all linear layers
```

**Custom Datasets**

`axolotl` supports [many dataset formats](https://github.com/OpenAccess-AI-Collective/axolotl#dataset). We recommend adding your custom dataset as a `.jsonl` file in the `data` folder and making the appropriate modifications to your config.

**Logging with Weights and Biases**

To track your training runs with Weights and Biases, add your `wandb` config information to your `config.yml`:

```yaml
wandb_project: code-7b-sql-output # set the project name
wandb_watch: gradients # track histograms of gradients
```

and set the `ALLOW_WANDB` environment variable to `true` when launching your training job:

```bash
ALLOW_WANDB=true modal run --detach src.train --config=... --data=...
```

### Multi-GPU training

We recommend [DeepSpeed](https://github.com/microsoft/DeepSpeed) for multi-GPU training, which is easy to set up. `axolotl` provides several default deepspeed JSON [configurations](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/deepspeed_configs) and Modal makes it easy to [attach multiple GPUs](https://modal.com/docs/guide/gpu#gpu-acceleration) of any type in code, so all you need to do is specify which of these configs you'd like to use.

First edit the DeepSpeed config in your `.yml`:

```yaml
deepspeed: /workspace/axolotl/deepspeed_configs/zero3_bf16.json
```

and then when you launch your training job,
set the `GPU_CONFIG` environment variable to the GPU configuration you want to use:

```bash
GPU_CONFIG=a100-80gb:4 modal run --detach src.train --config=... --data=...
```

### Finding and using your weights

You can find the results of all your runs via the CLI with

```bash
modal volume ls example-runs-vol
```

or view them in your [Modal dashboard](https://modal.com/storage).

You can browse the artifacts created by your training run with the following command, which is also printed out at the end of your training run in the logs:

```bash
modal volume ls example-runs-vol <run id>
# example: modal volume ls example-runs-vol axo-2024-04-13-19-13-05-0fb0
```

By default, the Modal `axolotl` trainer automatically merges the LoRA adapter weights into the base model weights.

The directory for a finished run will look like something this:

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

The LoRA adapters are stored in `lora-out`. The merged weights are stored in `lora-out/merged `. Note that many inference frameworks can only load the merged weights!

To run inference with a model from a past training job, you can specify the run name via the command line:

```bash
modal run -q src.inference --run-name=...
```

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
