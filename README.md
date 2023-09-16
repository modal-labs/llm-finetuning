# Fine-tune any Llama model in minutes

### Tired of prompt engineering? You've come to the right place.

This no-frills guide will take you from a dataset to a fine-tuned Llama model in the matter of minutes.

This repository is easy to tweak but comes ready to use as-is with all the recommended, start-of-the-art optimizations for fast results:

* *Fully-sharded data parallelism* so training scales optimally with multi-GPU
* *Parameter-efficient fine-tuning* via LoRa adapters for faster convergence
* *Gradient checkpointing* to reduce VRAM footprint, fit larger batches and get higher training throughput

The heavy lifting is done by the [our fork of the llama-recipes library](https://github.com/modal-labs/llama-recipes) (original [here](https://github.com/facebookresearch/llama-recipes)). Our fork patches support for Code Llama and an [open issue](https://github.com/facebookresearch/llama-recipes/issues/142) causing CUDA OOMs while saving LORA state dicts for 70B models.

Best of all, using Modal for fine-tuning means you never have to worry about infrastructure headaches like building images and provisioning GPUs. If a training script runs on Modal, it's repeatable and scalable enough to ship to production right away.

### Just one local dependency - a Modal account

1. Create a [Modal](https://modal.com/) account.
2. Install `modal` in your current Python virtual environment (`pip install modal`)
3. Set up a Modal token in your environment (`modal token new`)

### Training

To launch a training job, use:

```bash
modal run train.py --dataset sql_dataset.py --base chat7 --run-id chat7-sql
```

This example fine-tunes Llama 7B Chat to produce SQL queries (10k examples trained for 10 epochs in about 30 minutes). The base model nicknames used can be configured in `common.py`. 

Next, run inference to compare the results before/after training:
```bash
modal run inference.py --base chat7 --run-id chat7-sql --prompt '[INST] <<SYS>>
You are an advanced SQL assistant that uses this SQL table schema to generate a SQL query which answers the user question.
CREATE TABLE table_name_66 (points INTEGER, against VARCHAR, played VARCHAR)
<</SYS>>

What is the sum of Points when the against is less than 24 and played is less than 20? [/INST]'
```

*Coming soon: deploy a Text Generation Inference instance to scale up inference on your fine-tuned model to hundreds of tokens per second.*

### Bring your own dataset

Follow the example set by `sql_dataset.py` or `local_dataset.py` to import your own dataset. Then use

```bash
modal run validate_dataset.py --dataset new_dataset.py --base chat7 
```

to that validate your new script produces the desired training and test sets.

*Tip: ensure your training set is large enough to run a single step (i.e. contains at least `N_GPUS * batch_size` rows) to avoid `torch` errors.*
