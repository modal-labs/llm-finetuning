import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to train for only one epoch and truncate the dataset."""
    train_set_size = 1000
    val_set_size = 1000
    with open(config) as f:
        cfg = yaml.safe_load(f.read())
    cfg["val_set_size"] = val_set_size
    cfg["num_epochs"] = 1
    cfg.pop("eval_steps", None)  # Evaluate once at the end of the epoch
    with open(config, "w") as f:
        yaml.dump(cfg, f)

    with open(data) as f:
        data_truncated = f.readlines()[: train_set_size + val_set_size]
    with open(data, "w") as f:
        f.writelines(data_truncated)


if __name__ == "__main__":
    main()
