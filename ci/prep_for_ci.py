import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to train for only one epoch and truncate the dataset."""
    with open(config) as f:
        cfg = yaml.safe_load(f.read())

    if cfg["sample_packing"]:
        train_set_size = 2048
        num_epochs = 4
    elif "pythia" in cfg["base_model"]:
        train_set_size = 3200
        num_epochs = 1
    else:
        train_set_size = 1024
        num_epochs = 1
    val_set_size = 64

    cfg["val_set_size"] = val_set_size
    cfg["num_epochs"] = num_epochs
    cfg.pop("eval_steps", None)  # Evaluate once at the end of the epoch
    with open(config, "w") as f:
        yaml.dump(cfg, f)

    with open(data) as f:
        data_truncated = f.readlines()[: train_set_size + val_set_size]
    with open(data, "w") as f:
        f.writelines(data_truncated)


if __name__ == "__main__":
    main()
