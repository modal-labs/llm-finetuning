import datasets

from llama_recipes.datasets.utils import Concatenator


def get_custom_dataset(dataset_config, tokenizer, split):
    full_dataset = datasets.load_dataset(
        "json", data_files="./llama_train.jsonl", split="train"
    )

    # Since the dataset has no train/test split, we create one and select it
    dataset = full_dataset.train_test_split(
        train_size=0.95,
        test_size=0.05,
        seed=42,
    )["train" if split == dataset_config.train_split else "test"]
    if split == dataset_config.train_split:
        # HACK: Need sufficient rows to avoid some distributed processing
        # errors. Should really just get more data.
        N = 15  # Haven't probed to get this right yet
        dataset = datasets.Dataset.from_dict({"text": dataset["text"] * N})

    dataset = dataset.map(
        lambda x: tokenizer(x["text"]), remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True, batch_size=None)
    return dataset
