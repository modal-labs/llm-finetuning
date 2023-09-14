from modal import Mount, gpu

from common import stub, BASE_MODELS

@stub.function(
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
    mounts=[
        Mount.from_local_dir("./helpers", remote_path="/root"),
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

    print(config)

    for split in ["train", "test"]:
        dataset = get_custom_dataset(config, tokenizer, split)
        print(split, len(dataset))

        first_sample = tokenizer.decode(dataset[0]["input_ids"])
        print(f"first sample:\n{first_sample[:500]} ...")
        
        print(dataset[0]["input_ids"][:25])


@stub.function(
    gpu=gpu.A100(memory=80),
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
)
def completion(base: str, prompt: str, run_id: str = ""):
    from transformers import (
        LlamaForCausalLM,
        AutoTokenizer,
        pipeline,
    )

    model_name = BASE_MODELS[base]

    print("Running completion for prompt:")
    print(prompt)
    
    print("Loading base model", model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map="auto",
    )
    tokenizer.add_special_tokens({ "pad_token": "<PAD>" })

    generation_kwargs = dict(
        repetition_penalty=1.1
    )

    print("Generating without adapter ...")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **generation_kwargs)
    print(pipe(prompt)[0]["generated_text"])

    if run_id:
        print(f"Loading adapter {run_id=} ...")
        model.load_adapter(f"/results/{run_id}")

        print("Generating with adapter ...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        print(pipe(prompt)[0]["generated_text"])
