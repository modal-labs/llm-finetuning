# Optional stand-alone helper GUI to call the backend training functions.

import modal

from train import APP_NAME, VOLUME_CONFIG

stub = modal.Stub("example-axolotl-gui")

gradio_image = modal.Image.debian_slim().pip_install("gradio==4.5.0")

@stub.function(image=gradio_image, volumes=VOLUME_CONFIG, timeout=3600)
def gui(config_raw: str, data_raw: str):
    import gradio as gr
    import glob

    # Find the deployed business functions to call
    try:
        new = modal.Function.lookup(APP_NAME, "new")
        Inference = modal.Cls.lookup(APP_NAME, "Inference")
    except modal.exception.NotFoundError:
        raise Exception("Must first deploy training backend with `modal deploy train.py`.")

    def launch_training_job(config_yml, my_data_jsonl):
        run_id, handle = new.remote(config_yml, my_data_jsonl)
        result = (
            f"Started run {run_id} -> in folder /runs/{run_id}.\n\n"
            f"Follow training logs at https://modal.com/logs/call/{handle.object_id}\n"
        )
        print(result)
        return result

    def process_inference(model: str, input_text: str):
        text = f"Model: {model}\n{input_text}"
        yield text + "... (model loading)"
        try:
            for chunk in Inference(model.split("@")[-1]).completion.remote_gen(input_text):
                text += chunk
                yield text
        except Exception as e:
            return repr(e)

    def model_changed(model):
        return f"Model changed to: {model}"

    def get_model_choices():
        choices = [
            *glob.glob("/runs/*/lora-out/merged", recursive=True),
            *glob.glob("/pretrained/models--*/snapshots/*", recursive=True),
        ]
        choices = [f"{choice.split('/')[2]}@{choice}" for choice in choices]
        return gr.Dropdown(label="Select Model", choices=choices)

    with gr.Blocks() as interface:
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    config_input = gr.Code(label="config.yml", lines=20, value=config_raw)
                    data_input = gr.Code(label="my_data.jsonl", lines=20, value=data_raw)
                with gr.Column():
                    train_button = gr.Button("Launch training job")
                    train_output = gr.Markdown(label="Training details")
                    train_button.click(
                        launch_training_job,
                        inputs=[config_input, data_input],
                        outputs=train_output,
                    )

        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    model_dropdown = get_model_choices()
                    input_text = gr.Textbox(
                        label="Input Text (please include prompt manually)",
                        lines=10,
                        value="[INST] How do I deploy a Modal function? [/INST]",
                    )
                    inference_button = gr.Button("Run Inference")
                    refresh_button = gr.Button("Refresh")
                with gr.Column():
                    inference_output = gr.Textbox(label="Output", lines=20)
                    inference_button.click(
                        process_inference,
                        inputs=[model_dropdown, input_text],
                        outputs=inference_output,
                    )
            refresh_button.click(get_model_choices, inputs=None, outputs=model_dropdown)
            model_dropdown.change(
                model_changed, inputs=model_dropdown, outputs=inference_output
            )

        with gr.Tab("Files"):
            gr.FileExplorer(root="/runs")

    with modal.forward(8000) as tunnel:
        print("GUI available at", tunnel.url)
        interface.launch(
            quiet=True, show_api=False, server_name="0.0.0.0", server_port=8000
        )


@stub.local_entrypoint()
def main():
    with open("config.yml", "r") as config, open("my_data.jsonl", "r") as data:
        gui.remote(config.read(), data.read())
