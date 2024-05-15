from io import StringIO
import re
import sys

import pandas as pd

from modal import Volume


if __name__ == "__main__":
    with open(".last_run_name", "r") as f:
        run_name = f.read().strip()

    vol = Volume.lookup("example-runs-vol")
    contents = b""
    for chunk in vol.read_file(f"{run_name}/lora-out/README.md"):
        contents += chunk

    m = re.search(r"### Training results\n\n(.+?)#", contents.decode(), flags=re.DOTALL)
    if m is None:
        sys.exit("Could not parse training results from model card")
    else:
        results_text = m.group(1).strip().replace(" ", "")

    results = pd.read_table(StringIO(results_text), sep="|")
    train_loss = float(results["TrainingLoss"].iloc[-1])
    val_loss = float(results["ValidationLoss"].iloc[-1])

    # maximum loss for training, minimum loss for validation
    max_loss = 3e-2 if b"pythia" in contents else 2e-3  # pythia starts at higher loss
    min_loss = 0.2

    # mixtral training is not well-tuned, loosen learning requirement
    max_loss = max_loss * 10 if b"mixtral" in contents else max_loss

    print(f"Loss: {train_loss:.2f} (training), {val_loss:.2f} (validation)")
    sys.exit(train_loss > max_loss or val_loss < min_loss)
