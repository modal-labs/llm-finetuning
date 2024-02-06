import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to train for only one epoch and truncate the dataset."""
    with open(config) as f:
        cfg = yaml.safe_load(f.read())
    cfg["num_epochs"] = 1
    with open(config, "w") as f:
        yaml.dump(cfg, f)

    with open(data) as f:
        data_truncated = f.readlines()[:2000]
    with open(data, "w") as f:
        f.writelines(data_truncated)


if __name__ == "__main__":
    main()
