import subprocess


if __name__ == "__main__":

    with open(".last_run_name", "r") as f:
        run_name = f.read().strip()

    prompt = """[INST] Using the schema context below, generate a SQL query that answers the question.
CREATE TABLE head (age INTEGER)
How many heads of the departments are older than 56 ? [/INST] """

    p = subprocess.Popen(["modal", "run", "src.inference", "--run-folder", f"/runs/{run_name}", "--prompt", prompt], stdout=subprocess.PIPE)
    output = ""

    for line in iter(p.stdout.readline, b''):
        output += line.decode()
        print(line.decode(), end="")
    
    print("Asserting that the output contains the expected SQL query")
    assert b"[SQL] SELECT" in output and b"[/SQL]" in output
