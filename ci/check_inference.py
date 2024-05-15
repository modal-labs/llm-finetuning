import os
import subprocess


if __name__ == "__main__":
    prompt = """[INST] Using the schema context below, generate a SQL query that answers the question.
CREATE TABLE head (age INTEGER)
How many heads of the departments are older than 56 ? [/INST] """

    os.environ["INFERENCE_GPU_CONFIG"] = "h100:2"
    p = subprocess.Popen(
        ["modal", "run", "src.inference", "--prompt", prompt], stdout=subprocess.PIPE
    )
    output = ""

    for line in iter(p.stdout.readline, b""):
        output += line.decode()
        print(line.decode())

    print("Asserting that the output contains the expected SQL query")
    assert "[SQL] SELECT" in output and "[/SQL]" in output
