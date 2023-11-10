from pathlib import Path
from typing import Optional
from modal import Image, gpu, method

import subprocess
import os

from common import stub, BASE_MODELS, VOLUME_CONFIG

tgi_image = (
    Image.from_registry("ghcr.io/huggingface/text-generation-inference:1.0.3")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install("text-generation", "transformers>=4.33.0")
    .env(dict(HUGGINGFACE_HUB_CACHE="/pretrained"))
)


@stub.function(image=tgi_image, volumes=VOLUME_CONFIG, timeout=60 * 20)
def merge(run_id: str, commit: bool = False):
    from text_generation_server.utils.peft import download_and_unload_peft

    os.mkdir(f"/results/{run_id}/merged")
    subprocess.call(f"cp /results/{run_id}/*.* /results/{run_id}/merged", shell=True)

    print(f"Merging weights for fine-tuned {run_id=}.")
    download_and_unload_peft(f"/results/{run_id}/merged", None, False)

    if commit:
        print("Committing merged model permanently (can take a few minutes).")
        stub.results_volume.commit()


@stub.cls(
    image=tgi_image,
    gpu=gpu.A100(count=1, memory=40),
    allow_concurrent_inputs=100,
    volumes=VOLUME_CONFIG,
)
class Model:
    def __init__(self, base: str = "", run_id: str = ""):
        from text_generation import AsyncClient
        import socket
        import time

        model = f"/results/{run_id}/merged" if run_id else BASE_MODELS[base]

        if run_id and not os.path.isdir(model):
            merge.local(run_id)  # local = run in the same container

        launch_cmd = ["text-generation-launcher", "--model-id", model, "--port", "8000"]
        self.launcher = subprocess.Popen(launch_cmd, stdout=subprocess.DEVNULL)

        self.client = None
        while not self.client and self.launcher.returncode is None:
            try:
                socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
                self.client = AsyncClient("http://127.0.0.1:8000", timeout=60)
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(1.0)

        assert self.launcher.returncode is None

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.launcher.terminate()

    @method()
    async def generate(self, prompt: Optional[str] = None, max_new_tokens: int = 512):
        result = await self.client.generate(prompt, max_new_tokens=max_new_tokens)

        return result.generated_text


def parse_arg(x: str):
    import re

    if "</s>" in x:
        truncated = x.split("</s>")[0].strip()
    else:
        truncated = x.strip()
    match = re.findall(r"(CORRECT|INCORRECT)", truncated)
    if match:
        res = match[-1]
    else:
        res = None
    scores = {
        "CORRECT": 1,
        "INCORRECT": 0,
    }
    return {"value": res, "score": scores.get(res), "comment": x}


def get_examples(dataset_name):
    from langsmith import Client

    client = Client()
    return list(client.list_examples(dataset_name=dataset_name))


def get_chain(model):
    print("Made model")
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(
        "<s>[INST] <<SYS>>\n"
        "You are evaluating a predicted answer to a question."
        "You must mark it as CORRECT or INCORRECT, based on the reference answer.\n"
        "<</SYS>>\n\n"
        "<Question>\n{question}\n</Question>"
        "<Prediction>\n{prediction}\n</Prediction>"
        "<Reference>\n{reference}\n</Reference>\n"
        "[/INST]\n"
    )
    return prompt | (lambda val: model.generate.remote(val.to_string())) | parse_arg


def evaluate(chain, dataset_name):
    from langsmith.schemas import Example, Run
    from langsmith import EvaluationResult, RunEvaluator
    from langchain.smith import RunEvalConfig
    from langsmith import Client

    client = Client()

    class AbsDistEvaluator(RunEvaluator):
        def _evaluate_run(self, expected, predicted):
            if predicted is not None:
                return EvaluationResult(
                    key="abs_dist",
                    score=abs(expected - predicted),
                    value=predicted,
                )
            return EvaluationResult(
                key="abs_dist",
                comment=f"Predicted value is None, expected {expected}",
            )

        def evaluate_run(
            self, run: Run, example: Example | None = None
        ) -> EvaluationResult:
            expected = next(iter(example.outputs.values()))
            predicted = run.outputs.get("score") if run.outputs else None
            return self._evaluate_run(expected, predicted)

    class BinaryEvaluator(AbsDistEvaluator):
        def _evaluate_run(self, expected, predicted):
            expected = 1 if expected > 0.5 else 0
            if predicted is not None:
                return EvaluationResult(
                    key="bin_dist",
                    score=expected == predicted,
                    value=predicted,
                )
            return EvaluationResult(
                key="bin_dist",
                comment=f"Predicted value is None, expected {expected}",
            )

    client = Client()
    client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=chain,
        evaluation=RunEvalConfig(
            custom_evaluators=[BinaryEvaluator(), AbsDistEvaluator()]
        ),
    )


@stub.local_entrypoint()
def main(
    base: str,
    run_id: str = "",
    dataset: str = "evaluator-ft-eval-dataset",
):
    print(f"Running completion for dataset:\n{dataset} and run_id: {run_id}")

    model = Model(base, run_id)
    print("Getting chain")
    chain = get_chain(model)
    print("Evaluating")
    evaluate(chain, dataset)
