from io import StringIO
import re
import sys

import pandas as pd

from modal import Volume


if __name__ == "__main__":

    with open(".last_run_folder", "r") as f:
        run_folder = f.read().strip()

    vol = Volume.lookup("example-runs-vol")
    contents = b""
    for chunk in vol.read_file(f"{run_folder}/lora-out/README.md"):
        contents += chunk

    m = re.search(r"### Training results\n\n(.+?)#", contents.decode(), flags=re.DOTALL)
    if m is None:
        sys.exit("Could not parse training results from model card")
    else:
        results_text = m.group().replace(" ", "")

    results = pd.read_table(StringIO(results_text), sep="|")
    train_loss = results["TrainingLoss"].iloc[-1].astype(float)
    val_loss = results["ValidationLoss"].iloc[-1].astype(float)

    print("Loss: {train_loss:.2f} (training), {val_loss:.2f} (validation)")
    sys.exit(val_loss < 0.25)  # Arbitrary threshold
