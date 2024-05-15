import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to train for only one epoch and truncate the dataset."""
    with open(config) as f:
        cfg = yaml.safe_load(f.read())

    num_epochs = 50
    val_set_size = 0.5

    if cfg["base_model"] == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        num_epochs = 25  # mixtral training is slower and not well-tuned, cut early

    cfg["val_set_size"] = val_set_size
    cfg["num_epochs"] = num_epochs
    cfg["eval_steps"] = num_epochs
    cfg.pop("evals_per_epoch", None)  # incompatible with eval_steps
    cfg.pop("sample_packing", False)  # requires larger dataset

    with open(config, "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()
