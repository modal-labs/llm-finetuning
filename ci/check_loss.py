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

    print(f"Loss: {train_loss:.2f} (training), {val_loss:.2f} (validation)")
    sys.exit(val_loss > 0.4)  # Arbitrary threshold
