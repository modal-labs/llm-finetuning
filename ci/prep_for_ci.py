import click
import yaml


@click.command()
@click.option("--config")
@click.option("--data")
def main(config: str, data: str):
    """Set the config to train for only one epoch and truncate the dataset."""
    with open(config) as f:
        cfg = yaml.safe_load(f.read())
    cfg["sample_packing"] = False
    cfg["pad_to_sequence_len"] = False
    cfg["micro_batch_size"] = 32
    cfg["gradient_accumulation_steps"] = 1
    cfg["learing_rate"] = 0.0001
    cfg["num_epochs"] = 1
    with open(config, "w") as f:
        yaml.dump(cfg, f)

    with open(data) as f:
        data_truncated = f.readlines()[:1000]
    with open(data, "w") as f:
        f.writelines(data_truncated)


if __name__ == "__main__":
    main()
