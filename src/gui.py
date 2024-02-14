# Optional stand-alone helper GUI to call the backend training functions.

import modal

import webbrowser
import os

from .common import APP_NAME, VOLUME_CONFIG

stub = modal.Stub("example-axolotl-gui")
stub.q = modal.Queue.new()  # Pass back the URL to auto-launch

gradio_image = modal.Image.debian_slim().pip_install("gradio==4.5.0")


@stub.function(image=gradio_image, volumes=VOLUME_CONFIG, timeout=3600)
def gui(config_raw: str, data_raw: str):
    import gradio as gr
    import glob

    # Find the deployed business functions to call
    try:
        launch = modal.Function.lookup(APP_NAME, "launch")
        Inference = modal.Cls.lookup(APP_NAME, "Inference")
    except modal.exception.NotFoundError:
        raise Exception(
            "Must first deploy training backend with `modal deploy train.py`."
        )

    def jobs_table():
        VOLUME_CONFIG["/runs"].reload()

        md = "|Run|Checkpoint (steps)|Merged|Logs|\n|-|-|-|-|\n"
        for run in reversed(sorted(glob.glob("/runs/*"))):
            checkpoints = [
                int(path.split("-")[-1])
                for path in glob.glob(f"{run}/lora-out/checkpoint-*")
            ]
            last_checkpoint = max(checkpoints, default=0)
            merged = "✅" if glob.glob(f"{run}/lora-out/merged/*") else "..."
            try:
                with open(f"{run}/logs.txt") as f:
                    logs = f.read().strip()
            except FileNotFoundError:
                logs = "No logs link"
            md += "| {} | {} | {} | {} |\n".format(run, last_checkpoint, merged, logs)

        print(md)

        return md

    def launch_training_job(config_yml, my_data_jsonl):
        run_folder, handle = launch.remote(config_yml, my_data_jsonl)
        result = (
            f"Started training run in folder {run_folder}.\n\n"
            f"Follow training logs at https://modal.com/logs/call/{handle.object_id}\n"
        )
        print(result)
        return result

    def process_inference(model: str, input_text: str):
        text = f"Model: {model}\n{input_text}"
        yield text + "... (model loading)"
        try:
            for chunk in Inference(model.split("@")[-1]).completion.remote_gen(
                input_text
            ):
                text += chunk
                yield text
        except Exception as e:
            return repr(e)

    def model_changed(model):
        # Warms up a container with this model using an empty input
        # if model: list(Inference(model.split("@")[-1]).completion.remote_gen(""))

        # Show the config and data for this model
        try:
            run_folder = f"/runs/{model.split('@')[0]}"
            with (
                open(f"{run_folder}/config.yml", "r") as config,
                open(f"{run_folder}/my_data.jsonl", "r") as data,
            ):
                return config.read(), data.read()
        except (AttributeError, FileNotFoundError):
            return None, None

    def get_model_choices():
        VOLUME_CONFIG["/runs"].reload()
        VOLUME_CONFIG["/pretrained"].reload()
        choices = [
            *glob.glob("/runs/*/lora-out/merged", recursive=True),
            *glob.glob("/pretrained/models--*/snapshots/*", recursive=True),
        ]
        choices = [f"{choice.split('/')[2]}@{choice}" for choice in choices]
        return reversed(sorted(choices))

    with gr.Blocks() as interface:
        with gr.Tab("Train"):
            with gr.Accordion("Training summary"):
                train_status = gr.Markdown(label="Training status", value=jobs_table())

                refresh_button = gr.Button("Refresh", size="sm")
                refresh_button.click(jobs_table, outputs=[train_status])

            with gr.Row():
                with gr.Tab("Config (YAML)"):
                    config_input = gr.Code(
                        label="config.yml", lines=20, value=config_raw
                    )
                with gr.Tab("Data (JSONL)"):
                    data_input = gr.Code(
                        label="my_data.jsonl", lines=20, value=data_raw
                    )
                with gr.Column():
                    with gr.Group():
                        train_button = gr.Button(
                            "Launch training job", variant="primary"
                        )
                        train_output = gr.Markdown(label="Training details")

                    train_button.click(
                        launch_training_job,
                        inputs=[config_input, data_input],
                        outputs=train_output,
                    )

                    gr.FileExplorer(root="/runs")

        with gr.Tab("Inference"):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        model_dropdown = gr.Dropdown(
                            label="Select Model", choices=get_model_choices()
                        )
                        refresh_button = gr.Button("Refresh", size="sm")
                    with gr.Tab("Config (YAML)"):
                        model_config = gr.Code(label="config.yml", lines=20)
                    with gr.Tab("Data (JSONL)"):
                        model_data = gr.Code(label="my_data.jsonl", lines=20)

                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text (please include prompt manually)",
                        lines=10,
                        value="[INST] How do I deploy a Modal function? [/INST]",
                    )
                    inference_button = gr.Button(
                        "Run Inference", variant="primary", size="sm"
                    )
                    inference_output = gr.Textbox(label="Output", lines=20)

                    inference_button.click(
                        process_inference,
                        inputs=[model_dropdown, input_text],
                        outputs=inference_output,
                    )

            refresh_button.click(
                lambda: gr.update(choices=get_model_choices()),
                inputs=None,
                outputs=[model_dropdown],
            )
            model_dropdown.change(
                model_changed,
                inputs=model_dropdown,
                outputs=[model_config, model_data],
            )

    with modal.forward(8000) as tunnel:
        stub.q.put(tunnel.url)
        interface.launch(quiet=True, server_name="0.0.0.0", server_port=8000)


@stub.local_entrypoint()
def main():
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f"{parent}/config/mixtral.yml", "r") as cfg, open(
        f"{parent}/data/sqlqa.jsonl", "r"
    ) as data:
        handle = gui.spawn(cfg.read(), data.read())
    url = stub.q.get()
    print(f"GUI available at -> {url}\n")
    webbrowser.open(url)
    handle.get()
