from modal import gpu

from common import stub, BASE_MODELS

@stub.function(
    gpu=gpu.A100(memory=40),
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
        temperature=0.05,
        repetition_penalty=1.1,
        max_length=256,
    )

    print("=" * 20 + "Generating without adapter" + "=" * 20)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **generation_kwargs)
    print(pipe(prompt)[0]["generated_text"])

    if run_id:
        print(f"Loading adapter {run_id=} ...")
        model.load_adapter(f"/results/{run_id}")

        print("=" * 20 + "Generating with adapter" + "=" * 20)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            **generation_kwargs,
        )
        print(pipe(prompt)[0]["generated_text"])
