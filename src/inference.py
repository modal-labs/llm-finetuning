import time
import yaml
from pathlib import Path

import modal
from fastapi.responses import StreamingResponse

from .common import app, vllm_image, VOLUME_CONFIG

N_INFERENCE_GPU = 2

with vllm_image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid


def get_model_path_from_run(path: Path) -> Path:
    with (path / "config.yml").open() as f:
        return path / yaml.safe_load(f.read())["output_dir"] / "merged"


@app.cls(
    gpu=modal.gpu.H100(count=N_INFERENCE_GPU),
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=900,
)
class Inference:
    def __init__(self, run_name: str = "", run_dir: str = "/runs") -> None:
        self.run_name = run_name
        self.run_dir = run_dir

    @modal.enter()
    def init(self):
        if self.run_name:
            path = Path(self.run_dir) / self.run_name
            model_path = get_model_path_from_run(path)
        else:
            # Pick the last run automatically
            run_paths = list(Path(self.run_dir).iterdir())
            for path in sorted(run_paths, reverse=True):
                model_path = get_model_path_from_run(path)
                if model_path.exists():
                    break

        print("Initializing vLLM engine on:", model_path)

        engine_args = AsyncEngineArgs(
            model=model_path,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=N_INFERENCE_GPU,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _stream(self, input: str):
        if not input:
            return

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                request_output.outputs[0].text
                and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)

    @modal.method()
    async def completion(self, input: str):
        async for text in self._stream(input):
            yield text

    @modal.method()
    async def non_streaming(self, input: str):
        output = [text async for text in self._stream(input)]
        return "".join(output)

    @modal.web_endpoint()
    async def web(self, input: str):
        return StreamingResponse(self._stream(input), media_type="text/event-stream")


@app.local_entrypoint()
def inference_main(run_name: str = "", prompt: str = ""):
    if prompt:
        for chunk in Inference(run_name).completion.remote_gen(prompt):
            print(chunk, end="")
    else:
        prompt = input(
            "Enter a prompt (including the prompt template, e.g. [INST] ... [/INST]):\n"
        )
        print("Loading model ...")
        for chunk in Inference(run_name).completion.remote_gen(prompt):
            print(chunk, end="")
