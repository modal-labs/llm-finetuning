import json


def is_valid_jsonl(file_path):
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {line_number}: {e}")
                return False
    return True


file_path = "data/transcripts.jsonl"
if is_valid_jsonl(file_path):
    print("The file is in proper JSON Lines format.")
else:
    print("The file is not in proper JSON Lines format.")
