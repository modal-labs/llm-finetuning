import datasets

from llama_recipes.datasets.utils import Concatenator


B_INST, E_INST = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def format_text(row, tokenizer):
    text = (
        B_INST
        + B_SYS
        + "You are an advanced SQL assistant that uses this SQL table schema to generate a SQL query which answers the user question.\n"
        + row["context"]
        + E_SYS
        + row["question"]
        + E_INST
        + "[SQL] "
        + row["answer"]
        + " [/SQL]</s>"
    )

    return tokenizer(text)


def get_custom_dataset(dataset_config, tokenizer, split):
    full_dataset = datasets.load_dataset("b-mc2/sql-create-context", split="train")

    # Since the dataset has no train/test split, we create one and select it
    dataset = full_dataset.train_test_split(
        train_size=10000,
        test_size=200,
        seed=42,
    )["train" if split == dataset_config.train_split else "test"]

    dataset = dataset.map(
        lambda x: format_text(x, tokenizer),
        remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True)

    return dataset
