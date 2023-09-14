import datasets

from llama_recipes.datasets.utils import Concatenator

def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("json", data_files="./modal_docs.jsonl", split="train")

    # Since the dataset has no train/test split, we create one and select it
    dataset = dataset.train_test_split(
        train_size=0.9,
        test_size=0.1,
        seed=42,
    )[split]

    dataset = dataset.map(
        lambda x: tokenizer(x["text"]),
        remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True)

    return dataset
