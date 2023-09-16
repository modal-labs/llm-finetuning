from modal import gpu, method

from common import stub, BASE_MODELS


@stub.cls(
    gpu=gpu.A100(count=1, memory=40),
    volumes={
        "/pretrained": stub.pretrained_volume,
        "/results": stub.results_volume,
    },
)
class Model:
    def __init__(self, base: str):
        from transformers import LlamaForCausalLM, AutoTokenizer
        import logging

        model_name = BASE_MODELS[base]
        print("Loading base model", model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        # Hide annoying warnings
        logging.getLogger("optimum.bettertransformer").setLevel(logging.CRITICAL)

        self.model = self.model.to_bettertransformer()
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @method()
    def generate(self, prompt: str, run_id: str = "", verbose: bool = False):
        from transformers import GenerationConfig, TextStreamer
        from peft import PeftModel

        if run_id:
            print("=" * 20 + "Generating with adapter" + "=" * 20)
            print(f"Loading adapter {run_id=}.")
            self.model = PeftModel.from_pretrained(
                self.model,
                f"/results/{run_id}",
                is_trainable=False,
            )
        elif verbose:
            print("=" * 20 + "Generating without adapter" + "=" * 20)

        input_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids
        output_tokens = self.model.generate(
            inputs=input_tokens.to(self.model.device),
            streamer=TextStreamer(self.tokenizer, skip_prompt=True)
            if verbose
            else None,
            generation_config=GenerationConfig(max_new_tokens=512),
        )[0]

        if run_id:
            self.model = self.model.unload()

        return self.tokenizer.decode(
            output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


@stub.local_entrypoint()
def main(base: str, prompt: str, run_id: str = "", map_sz: int = 0):
    print(f"Running completion for prompt:\n{prompt}")

    Model(base).generate.remote(prompt, verbose=True)

    if run_id:
        Model(base).generate.remote(prompt, run_id, verbose=True)

    if map_sz > 1:
        for _output in Model(base).generate.map([prompt] * map_sz):
            pass
