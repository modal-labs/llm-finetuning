# Optional stand-alone helper GUI to call the backend training functions.

import modal

from train import VOLUME_CONFIG
import time

stub = modal.Stub("example-axolotl-gui")

gradio_image = modal.Image.debian_slim().pip_install("gradio==4.5.0")

vllm_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to 10/16/23
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@651c614aa43e497a2e2aab473493ba295201ab20"
    )
)

@stub.cls(gpu="A100", image=vllm_image, volumes=VOLUME_CONFIG, allow_concurrent_inputs=60,  container_idle_timeout=120)
class Model:
    def __init__(self, model_path: str) -> None:
        print(model_path)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(model=model_path, gpu_memory_utilization=0.95)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def completion(self, input: str):
        if not input:
            return

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            presence_penalty=0.8,
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
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)


@stub.function(image=gradio_image, volumes=VOLUME_CONFIG, timeout=3600)
def gui():
    import gradio as gr
    import glob

    def process_train_files(config_yml, my_data_jsonl):
        # Processing logic for training files
        return "Training files processed."

    def process_inference(model: str, input_text: str):
        text = f"Model: {model}\n{input_text}"
        yield text + "... (model loading)"
        for chunk in Model(model.split("@")[-1]).completion.remote_gen(input_text):
            text += chunk
            yield text

    def model_changed(model):
        return f"Model changed to: {model}"

    def get_model_choices():
        choices = [
            *glob.glob("/runs/*/lora-out/merged", recursive=True),
            *glob.glob("/pretrained/models--*/snapshots/*", recursive=True)
        ]
        choices = [f"{choice.split('/')[2]}@{choice}" for choice in choices]
        return gr.Dropdown(label="Select Model", choices=choices)

    with gr.Blocks() as interface:
        with gr.Tab("Train"):
            with gr.Column():
                config_input = gr.Textbox(label="config.yml", lines=10, placeholder="Enter YAML content here")
                data_input = gr.Textbox(label="my_data.jsonl", lines=10, placeholder="Enter JSON Lines content here")
            with gr.Column():
                train_button = gr.Button("Launch training job")
                train_output = gr.Textbox(label="Training details")
                train_button.click(process_train_files, inputs=[config_input, data_input], outputs=train_output)

        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = get_model_choices()
                    input_text = gr.Textbox(label="Input Text", lines=10, placeholder="Enter text here")
                    inference_button = gr.Button("Run Inference")
                with gr.Column():
                    inference_output = gr.Textbox(label="Output", lines=50)
                    inference_button.click(process_inference, inputs=[model_dropdown, input_text], outputs=inference_output)
            refresh_button = gr.Button("Refresh")
            refresh_button.click(get_model_choices, inputs=None, outputs=model_dropdown)
            model_dropdown.change(model_changed, inputs=model_dropdown, outputs=inference_output)

        with gr.Tab("Files"):
            gr.FileExplorer(root="/runs")


    with modal.forward(8000) as tunnel:
        print("GUI available at", tunnel.url)
        interface.launch(quiet=True, show_api=False, server_name="0.0.0.0", server_port=8000)

@stub.local_entrypoint()
def main():
    gui.remote()