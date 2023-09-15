from modal import gpu, method

from common import stub, BASE_MODELS, GPU_MEM


@stub.cls(
    gpu=gpu.A100(memory=GPU_MEM),
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
)
class Model:
    def __init__(self, base: str):
        from transformers import LlamaForCausalLM, AutoTokenizer

        print("Loading base model", BASE_MODELS[base])
        self.model = LlamaForCausalLM.from_pretrained(
            BASE_MODELS[base],
            torch_dtype="auto",
            device_map="auto",
        ).to_bettertransformer()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODELS[base])
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @method()
    def generate(self, prompt: str, run_id: str = ""):
        from transformers import pipeline

        if run_id:
            print(f"Loading adapter {run_id=}.")
            self.model.load_adapter(f"/results/{run_id}")

        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=True,
            temperature=0.05,
            repetition_penalty=1.1,
            max_length=256,
        )
        print(pipe(prompt)[0]["generated_text"])

        if run_id:
            print("Disabling adapter.")
            self.model.disable_adapters()


@stub.local_entrypoint()
def main(base: str, prompt: str, run_id: str = ""):
    print(f"Running completion for prompt:\n{prompt}")

    print("=" * 20 + "Generating without adapter" + "=" * 20)
    Model(base).generate.remote(prompt)

    if run_id:
        print("=" * 20 + "Generating with adapter" + "=" * 20)
        Model(base).generate.remote(prompt, run_id)
