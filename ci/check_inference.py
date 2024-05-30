import json
import subprocess
from pathlib import Path


if __name__ == "__main__":
    dataset = "sqlqa.subsample.jsonl"
    dataset_path = Path(__file__).parent.parent / "data" / dataset
    example = json.loads(dataset_path.read_text().strip().split("\n")[0])

    prompt = "[INST] Using the schema context below, generate a SQL query that answers the question."
    prompt += f"\n{example['context']}"
    prompt += f"\n{example['question']}"
    prompt += " [/INST]"

    p = subprocess.Popen(
        ["modal", "run", "src.inference", "--prompt", prompt],
        stdout=subprocess.PIPE,
    )
    output = ""

    for line in iter(p.stdout.readline, b""):
        output += line.decode()
        print(line.decode())

    print("Asserting that the output contains the expected format")
    assert "[SQL]" in output and "SELECT" in output and " [/SQL]" in output
