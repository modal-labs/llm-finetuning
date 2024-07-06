import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to run for many epochs to test overfitting."""
    with open(config) as f:
        cfg = yaml.safe_load(f.read())

    cfg["wandb_project"] = "ci-llm-finetuning-overfit-sqlqa"
    cfg["seed"] = 117  # always set a seed
    cfg["save_strategy"] = "no"
    # turn off regularization
    cfg["lora_dropout"] = 0
    cfg["weight_decay"] = 0

    val_set_size = 0.5

    num_epochs = 25
    if "CodeLlama-7b" in cfg["base_model"]:
        num_epochs = num_epochs + 10
    elif "pythia-1.4b" in cfg["base_model"]:
        num_epochs = num_epochs * 3

    cfg["val_set_size"] = val_set_size
    cfg["num_epochs"] = num_epochs
    cfg["eval_steps"] = num_epochs // 5
    cfg.pop("evals_per_epoch", None)  # incompatible with eval_steps
    cfg.pop("sample_packing", False)  # requires larger dataset

    with open(config, "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()
