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

* `new` prepares a new folder in the `/runs` volume with the training config and data for a new training job. It also ensures the base model is downloaded from HuggingFace.
* `train` takes a prepared folder and performs the training job using the config and data.
* `Inference.completion` can spawn a [vLLM](https://modal.com/docs/examples/vllm_inference#fast-inference-with-vllm-mistral-7b) inference container for any pre-trained or fine-tuned model from a previous training job.

The rest of the code are helpers for _calling_ these three functions. There are two main ways to train:

* Use the GUI to familiarize with the system (recommended for new fine-tuners!)
* Use CLI commands (recommended for power users)

### Using the GUI

Deploy the training backend to expose the business functions (`new`, `train` and `completion`), then run the Gradio GUI.

```bash
modal deploy train.py
modal run gui.py
```

The `*.modal.tunnel` link from the latter will take you to the Gradio GUI. There will be three tabs: launch training runs, test out trained models and explore the files on the volume.

The difference between deploying an app and running an ephemeral app is that deployed apps will remain ready on the cloud for any invocations, whereas ephemeral apps will shut down once your local command exits. This lets your training jobs (that run on the deployed app) continue without your laptop; but makes sure you are not wasting resources hosting a GUI when not at your laptop.

### Using the CLI

Test out a simple training job with:

```bash
modal run train.py
```

Training depends on two files: `axolotl-mount/config.yml` and `axolotl-mount/my_data.jsonl`. The folder is automatically [mounted](https://modal.com/docs/guide/local-data#mounting-directories) to a remote container. When you make local changes to either of these files, they will be reflected in the next training run.

The default configuration fine-tunes CodeLlama Instruct 7B to understand Modal documentation.

produce SQL queries (10k examples trained for 10 epochs in about 30 minutes). The base model nicknames used can be configured in `common.py` and are used to define which model is being trained.

Next, run inference to compare the results before/after training:
```bash
modal run inference.py --base chat7 --run-id chat7-sql --prompt '[INST] <<SYS>>
You are an advanced SQL assistant that uses this SQL table schema to generate a SQL query which answers the user question.
CREATE TABLE table_name_66 (points INTEGER, against VARCHAR, played VARCHAR)
<</SYS>>

What is the sum of Points when the against is less than 24 and played is less than 20? [/INST]'
```

Add `--batch 10000` to scale up seamlessly to dozens of GPUs for effortless parallelism as we complete 10000 prompts.

<img width="874" alt="Screenshot 2023-09-16 at 1 29 39 AM" src="https://github.com/modal-labs/llama-finetuning/assets/8001209/d35bb956-dca2-4cc4-bb42-1e1372650481">

### Fine-tuning on other models

You can train it using other models, including the 70B variants. In order for that to work, you need to lower the batch size.

```bash
modal run train.py --dataset sql_dataset.py --base chat70 --run-id chat70-sql --batch-size 4
```

### Bring your own dataset

Follow the example set by `sql_dataset.py` or `local_dataset.py` to import your own dataset. Then use

```bash
modal run validate_dataset.py --dataset new_dataset.py --base chat7 
```

to that validate your new script produces the desired training and test sets.

*Tip: ensure your training set is large enough to run a single step (i.e. contains at least `N_GPUS * batch_size` rows) to avoid `torch` errors.*
