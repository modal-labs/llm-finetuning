from modal import Mount, gpu

from common import stub, BASE_MODELS

@stub.function(
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
    mounts=[
        Mount.from_local_dir("./datasets", remote_path="/root"),
    ],
)
def dataset(base: str = "chat7", file: str = "local_dataset.py"):
    from llama_recipes.utils.dataset_utils import get_custom_dataset
    from llama_recipes.configs.datasets import custom_dataset
    from llama_recipes.utils.config_utils import update_config

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODELS[base])
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    config = custom_dataset()
    update_config(config, file=file)

    BLOCK = "=" * 20

    for split in [config.train_split, config.test_split]:
        dataset = get_custom_dataset(config, tokenizer, split)
        print(f"{split}: {len(dataset)} sequences")

        sample = tokenizer.decode(dataset[0]["input_ids"])[:500]
        print(f"{BLOCK} Sample {BLOCK}\n{sample} ...")
        print(f"{BLOCK} Tokens {BLOCK}\n{dataset[0]['input_ids'][:25]} ...\n")
