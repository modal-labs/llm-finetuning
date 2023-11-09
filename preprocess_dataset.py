import json

from langsmith import Client

client = Client()
# TODO: List the public dataset once shared
examples = list(client.list_examples(dataset_name="mistral-finetuning-v0"))


def format_prompt(question, prediction, reference):
    system_prompt = """You are evaluating a predicted answer to a question.\
 You must mark it as CORRECT or INCORRECT, based on the reference answer."""
    return (
        f"<s>[INST] <<SYS>>\n"
        f"{system_prompt.strip()}\n"
        f"<</SYS>>\n\n"
        f"<Question>\n{question}\n</Question>"
        f"<Prediction>\n{prediction}\n</Prediction>"
        f"<Reference>\n{reference}\n</Reference>\n"
        f"[/INST]\n"
    )


def format_data(example):
    prediction = example.inputs["pred"]
    reference = example.inputs["ref"]
    question = example.inputs["question"]
    answer = example.inputs["answer"]
    prompt = format_prompt(question, prediction, reference)
    return {"text": f"{prompt}{answer}\n</s>"}


instructions = [format_data(example) for example in examples]
with open("./datasets/llama_train.jsonl", "w") as f:
    for inst in instructions:
        f.write(f"{json.dumps(inst)}\n")
