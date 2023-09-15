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

        model_name = BASE_MODELS[base]
        print("Loading base model", model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        ).to_bettertransformer()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @method()
    def generate(self, prompt: str, run_id: str = ""):
        from transformers import pipeline, TextStreamer

        if run_id:
            print("=" * 20 + "Generating with adapter" + "=" * 20)
            print(f"Loading adapter {run_id=}.")
            self.model.load_adapter(f"/results/{run_id}")
        else:
            print("=" * 20 + "Generating without adapter" + "=" * 20)

        output = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            streamer=TextStreamer(self.tokenizer),
            do_sample=True,
            temperature=0.05,
            repetition_penalty=1.1,
            max_length=256,
        )(prompt)

        if run_id:
            self.model.disable_adapters()
        
        return output[0]["generated_text"]


@stub.local_entrypoint()
def main(base: str, prompt: str, run_id: str = ""):
    print(f"Running completion for prompt:\n{prompt}")

    Model(base).generate.remote(prompt)
    if run_id:
        Model(base).generate.remote(prompt, run_id)
