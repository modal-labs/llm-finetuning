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

    cfg["val_set_size"] = val_set_size
    cfg["num_epochs"] = num_epochs
    cfg["eval_steps"] = num_epochs

    with open(config, "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()
